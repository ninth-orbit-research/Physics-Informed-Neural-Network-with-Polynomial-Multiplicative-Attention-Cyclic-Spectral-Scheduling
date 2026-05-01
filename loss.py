"""
CFRP-PINN: Physics Loss Module
================================
Implements the micromechanical physics constraints that regularise the
neural network.  All computations are differentiable and numerically
stable (clamped denominators, log-space Weibull, masked integrals).

Physics summary
---------------
Two composite properties are predicted:

  σ_HT  — ultimate tensile strength (rule of mixtures + Kelly–Tyson)
  E_HT  — Young's modulus (Halpin–Tsai model)

The physics residual is:
    L_phys = MSE(σ̂_HT, σ_HT^theory) + MSE(Ê_HT, E_HT^theory)

Efficiency Factors
------------------
χ₁  (length efficiency factor, Kelly–Tyson):
    Accounts for the fact that fibres shorter than 2·L_crit transfer
    stress sub-optimally.  Computed as the Weibull-weighted average of
    ζ(L), where ζ = L/(2L_c) for L ≤ L_c and 1 - L_c/(2L) otherwise.

χ₂  (orientation efficiency factor):
    The fibre orientation distribution is modelled as a double-Gaussian
    on [0, π/2].  χ₂ = ∫h(θ)cos(θ)dθ · ∫h(θ)(cos³θ - ν sin²θ cosθ)dθ

L̄_n (number-average fibre length):
    Weibull-weighted mean length, used to compute the aspect ratio ξ.

Progressive Physics Weighting
------------------------------
λ(epoch) = λ_max · (1 − exp(−epoch / τ))

This ramp prevents the physics residual from dominating early training
when the data-fit term is still poorly conditioned.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ─────────────────────────────────────────────────────────────
#  Numerical integration helpers
# ─────────────────────────────────────────────────────────────

def trapz(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Vectorised trapezoidal integration.

    Args:
        y: Integrand values, shape (..., N).
        x: Quadrature nodes, shape (N,).  Must be 1-D.

    Returns:
        Integral approximation, shape (...,).
    """
    dx = x[1:] - x[:-1]                         # (N-1,)
    return torch.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, dim=-1)


# ─────────────────────────────────────────────────────────────
#  Efficiency Factors
# ─────────────────────────────────────────────────────────────

class EfficiencyFactors:
    """
    Compute χ₁, χ₂, and L̄_n via numerical quadrature.

    All integrals are evaluated on fixed grids and batched over the
    sample dimension.  Clamping and log-space arithmetic prevent NaN
    propagation through the Weibull PDF.

    Args:
        NL (int): Number of quadrature nodes along the length axis.
        NT (int): Number of quadrature nodes along the angle axis.
    """

    def __init__(self, NL: int = 256, NT: int = 256):
        self.NL = NL
        self.NT = NT

    # ── Weibull PDF ───────────────────────────────────────────

    @staticmethod
    def weibull_pdf(L: torch.Tensor,
                    a: torch.Tensor,
                    b: torch.Tensor) -> torch.Tensor:
        """
        Two-parameter Weibull PDF evaluated in log-space for stability.

        f(L; a, b) = (b/a) · (L/a)^{b-1} · exp[−(L/a)^b]

        Log-form:
            log f = log(b/a) + (b−1)·log(L/a) − (L/a)^b

        Args:
            L: Length values, shape (B, NL).
            a: Scale parameter, shape (B, NL).
            b: Shape parameter, shape (B, NL).

        Returns:
            PDF values, shape (B, NL).  Clamped to ≥ 0 by construction.
        """
        eps = 1e-12
        L = torch.clamp(L, min=eps)
        a = torch.clamp(a, min=eps)
        b = torch.clamp(b, min=eps)
        log_f = (
            torch.log(b / a)
            + (b - 1.0) * torch.log(L / a)
            - (L / a) ** b
        )
        return torch.exp(log_f)

    # ── χ₁ : length efficiency factor ────────────────────────

    def chi1(self,
             a: torch.Tensor,
             b: torch.Tensor,
             Lmin: torch.Tensor,
             Lmax: torch.Tensor,
             Lc: torch.Tensor) -> torch.Tensor:
        """
        Length efficiency factor (Kelly–Tyson).

        χ₁ = ∫_{Lmin}^{Lmax} f_W(L) · ζ(L) dL

        where:
            ζ(L) = L / (2·L_c)         if L ≤ L_c
                 = 1 − L_c / (2·L)     if L > L_c

        Args:
            a, b, Lmin, Lmax, Lc: Batched tensors, shape (B,).

        Returns:
            χ₁ values, shape (B,).
        """
        L_grid = torch.linspace(1e-6, 10.0, self.NL, device=a.device)
        L = L_grid[None, :].expand(a.size(0), -1)      # (B, NL)

        pdf = self.weibull_pdf(L, a[:, None], b[:, None])

        # Mask to [Lmin, Lmax] window
        mask = (L >= Lmin[:, None]) & (L <= Lmax[:, None])
        pdf  = pdf * mask.float()

        Lc_safe = torch.clamp(Lc[:, None], min=1e-6)
        zeta = torch.where(
            L <= Lc[:, None],
            L / (2.0 * Lc_safe),
            1.0 - Lc_safe / (2.0 * torch.clamp(L, min=1e-6))
        )

        return trapz(pdf * zeta, L_grid)

    # ── χ₂ : orientation efficiency factor ───────────────────

    def chi2(self,
             a1: torch.Tensor, b1: torch.Tensor, g1: torch.Tensor,
             a2: torch.Tensor, b2: torch.Tensor, g2: torch.Tensor,
             nu: torch.Tensor) -> torch.Tensor:
        """
        Orientation efficiency factor.

        Fibre orientation distribution h(θ) is a normalised mixture
        of two Gaussians on [0, π/2]:

            h(θ) ∝ a1·exp[−((θ−β₁)/γ₁)²] + a2·exp[−((θ−β₂)/γ₂)²]

        χ₂ = [ ∫h(θ)·cosθ dθ ] · [ ∫h(θ)·(cos³θ − ν·sin²θ·cosθ) dθ ]

        Args:
            a1,b1,g1: First Gaussian amplitude, mean, std (B,).
            a2,b2,g2: Second Gaussian amplitude, mean, std (B,).
            nu:       Poisson's ratio (B,).

        Returns:
            χ₂ values, shape (B,).
        """
        th_grid = torch.linspace(0, math.pi / 2, self.NT, device=a1.device)
        th = th_grid[None, :]

        g1 = torch.clamp(g1, min=1e-3)[:, None]
        g2 = torch.clamp(g2, min=1e-3)[:, None]

        h  = (
            a1[:, None] * torch.exp(-((th - b1[:, None]) / g1) ** 2)
            + a2[:, None] * torch.exp(-((th - b2[:, None]) / g2) ** 2)
        )

        # Normalise distribution
        Z  = torch.clamp(trapz(h, th_grid), min=1e-12)[:, None]
        h  = h / Z

        c, s = torch.cos(th), torch.sin(th)

        I1 = trapz(h * c, th_grid)                                   # (B,)
        I2 = trapz(h * (c ** 3 - nu[:, None] * s ** 2 * c), th_grid) # (B,)
        return I1 * I2

    # ── L̄_n : number-average fibre length ────────────────────

    def mean_length(self,
                    a: torch.Tensor,
                    b: torch.Tensor,
                    Lmin: torch.Tensor,
                    Lmax: torch.Tensor) -> torch.Tensor:
        """
        Number-average fibre length.

            L̄_n = ∫_{Lmin}^{Lmax} L · f_W(L) dL  /  ∫_{Lmin}^{Lmax} f_W(L) dL

        Args:
            a, b, Lmin, Lmax: Batched tensors, shape (B,).

        Returns:
            L̄_n values, shape (B,).
        """
        L_grid = torch.linspace(1e-6, 10.0, self.NL, device=a.device)
        L = L_grid[None, :].expand(a.size(0), -1)

        pdf  = self.weibull_pdf(L, a[:, None], b[:, None])
        mask = (L >= Lmin[:, None]) & (L <= Lmax[:, None])
        pdf  = pdf * mask.float()

        Z    = torch.clamp(trapz(pdf, L_grid), min=1e-12)
        return trapz(pdf * L / Z[:, None], L_grid)


# ─────────────────────────────────────────────────────────────
#  Physics Loss
# ─────────────────────────────────────────────────────────────

class PhysicsLoss(nn.Module):
    """
    Physics-informed loss combining data MSE and micromechanical residuals.

    Total loss:
        L_total = L_data + λ(epoch) · L_phys

    where:
        L_data  = MSE(ŷ, y)
        L_phys  = MSE(σ̂_HT^denorm, σ_HT^theory)
                + MSE(Ê_HT^denorm,  E_HT^theory)

    Theoretical targets:

        σ_HT = χ₁ · χ₂ · σ_f · w_f + (1 − w_f) · σ_m

        ξ    = 2 · L̄_n / D                         (aspect ratio)
        η    = (E_f − E_m) / (E_f + ξ · E_m)       (Halpin–Tsai η)
        E_HT = E_m · (1 + ξ · η · w_f)             (Halpin–Tsai)
                   / (1 − η · w_f)

    Args:
        use_progressive (bool): Enable progressive λ ramp-up.
        lambda_max      (float): Maximum physics weight.
        tau             (int): Time constant (epochs) for progressive ramp.
        NL, NT          (int): Quadrature grid sizes.
    """

    def __init__(self,
                 use_progressive: bool = True,
                 lambda_max: float = 0.5,
                 tau: int = 50,
                 NL: int = 256,
                 NT: int = 256):
        super().__init__()
        self.use_progressive = use_progressive
        self.lambda_max      = lambda_max
        self.tau             = tau
        self.eff             = EfficiencyFactors(NL=NL, NT=NT)
        self.epoch: int      = 0

    def update_epoch(self, e: int):
        """Call at the start of each epoch."""
        self.epoch = e

    def lambda_phys(self) -> float:
        """Current physics weight λ(epoch)."""
        if self.use_progressive:
            return self.lambda_max * (1.0 - math.exp(-self.epoch / max(self.tau, 1)))
        return self.lambda_max

    @staticmethod
    def _denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation: x_physical = x_norm * std + mean."""
        return x * std.to(x.device) + mean.to(x.device)

    def forward(self,
                yhat: torch.Tensor,
                y: torch.Tensor,
                x: torch.Tensor,
                ms: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the total loss.

        Args:
            yhat: Normalised predictions, shape (B, 2).
            y:    Normalised targets,     shape (B, 2).
            x:    Normalised inputs,      shape (B, 18).
            ms:   Dict with keys x_mean, x_std, y_mean, y_std.

        Returns:
            Scalar total loss tensor.
        """
        # ── Data loss (normalised space) ──────────────────────
        loss_data = F.mse_loss(yhat, y)

        # ── De-normalise for physics residual ─────────────────
        yhat_p = self._denorm(yhat, ms['y_mean'], ms['y_std'])   # (B, 2)
        x_p    = self._denorm(x,    ms['x_mean'], ms['x_std'])   # (B, 18)

        # ── Unpack physical inputs ─────────────────────────────
        w_f   = x_p[:, 0]
        a_w   = x_p[:, 1]    # Weibull scale
        b_w   = x_p[:, 2]    # Weibull shape
        Lmin  = x_p[:, 3]
        Lmax  = x_p[:, 4]
        Lc    = x_p[:, 5]
        D     = x_p[:, 6]
        al1   = x_p[:, 7];  b1 = x_p[:, 8];  g1 = x_p[:, 9]
        al2   = x_p[:, 10]; b2 = x_p[:, 11]; g2 = x_p[:, 12]
        E_m   = x_p[:, 13]
        sig_m = x_p[:, 14]
        E_f   = x_p[:, 15]
        sig_f = x_p[:, 16]
        nu    = x_p[:, 17]

        # ── Efficiency factors ────────────────────────────────
        chi1 = self.eff.chi1(a_w, b_w, Lmin, Lmax, Lc)
        chi2 = self.eff.chi2(al1, b1, g1, al2, b2, g2, nu)
        Ln   = self.eff.mean_length(a_w, b_w, Lmin, Lmax)

        # ── Strength: modified rule of mixtures ───────────────
        sigma_th = chi1 * chi2 * sig_f * w_f + (1.0 - w_f) * sig_m

        # ── Stiffness: Halpin–Tsai ────────────────────────────
        xi  = 2.0 * Ln / torch.clamp(D, min=1e-6)
        eta = (E_f - E_m) / torch.clamp(E_f + xi * E_m, min=1e-6)
        E_th = (
            E_m * (1.0 + xi * eta * w_f)
            / torch.clamp(1.0 - eta * w_f, min=1e-6)
        )

        # ── Physics residual ──────────────────────────────────
        loss_phys = (
            F.mse_loss(yhat_p[:, 0], sigma_th)
            + F.mse_loss(yhat_p[:, 1], E_th)
        )

        lam = self.lambda_phys()
        return loss_data + lam * loss_phys, {
            'loss_data':  loss_data.item(),
            'loss_phys':  loss_phys.item(),
            'lambda':     lam,
        }
