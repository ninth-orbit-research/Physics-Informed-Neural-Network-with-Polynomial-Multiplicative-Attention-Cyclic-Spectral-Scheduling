"""
CFRP-PINN: Core Neural Architecture
====================================
Implements three novel architectural contributions:

  1. **PMA** — Polynomial Multiplicative Attention
     A parameter-free nonlinearity that avoids activation functions entirely.
     Given a hidden vector z split into three chunks [z1, z2, z3], it returns
     [z1, z2 * z3], enabling multiplicative interactions (up to degree-2
     polynomial) without any learned gate weights. This preserves gradient
     flow while adding expressiveness beyond ReLU/GELU families.

  2. **CSS** — Cyclic Spectral Scheduling
     A custom backward-pass hook that applies band-specific learning-rate
     scaling in Fourier space. At each step, one spectral band is "active"
     (full gradient) while the remaining bands are attenuated. Bands rotate
     cyclically every T epochs, preventing the network from converging too
     fast in high-frequency directions before low-frequency structure is
     established — effectively combating spectral bias.

  3. **BranchNetwork + MHA Fusion**
     Each of the 18 input features gets its own dedicated branch encoder,
     then cross-feature dependencies are captured by Multi-Head Attention
     over the 18 branch embeddings. This allows the model to learn feature
     interaction patterns dynamically rather than relying on fixed concat.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  Polynomial Multiplicative Attention (PMA)
# ─────────────────────────────────────────────────────────────

class PMA(nn.Module):
    """
    Polynomial Multiplicative Attention (PMA).

    Splits the last dimension of the input into 3 equal chunks,
    then returns the concatenation of [chunk1, chunk2 * chunk3].

    This creates a multiplicative "gate" without any learned parameters:
        output = [z₁  |  z₂ ⊙ z₃]

    where ⊙ is element-wise multiplication.  The output dimension is
    (2/3) * d_in, so the caller must account for this when sizing
    subsequent linear layers.

    Mathematical motivation:
        - A standard two-layer MLP with ReLU approximates piece-wise
          linear functions.
        - PMA implicitly spans degree-2 polynomial terms (z_i * z_j),
          giving the network the power of a quadratic feature map
          without an explicit expansion.
        - Gradient norm through the multiplicative path is proportional
          to the magnitude of the counterpart chunk, not a fixed slope,
          which helps avoid dying-neuron pathologies.

    Args:
        d_h (int): Input hidden dimension.  Must be divisible by 3.

    Example::

        pma = PMA(d_h=96)
        z   = torch.randn(32, 96)
        out = pma(z)          # → (32, 64)
    """

    def __init__(self, d_h: int):
        super().__init__()
        assert d_h % 3 == 0, f"PMA requires d_h divisible by 3, got {d_h}"
        self.chunk   = d_h // 3
        self.out_dim = 2 * self.chunk   # caller uses this to size next linear

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z1 = z[..., : self.chunk]                   # identity path
        z2 = z[..., self.chunk : 2 * self.chunk]    # multiplicand
        z3 = z[..., 2 * self.chunk : 3 * self.chunk]  # multiplier
        return torch.cat([z1, z2 * z3], dim=-1)


# ─────────────────────────────────────────────────────────────
#  Cyclic Spectral Scheduling (CSS)
# ─────────────────────────────────────────────────────────────

class _CSSFunction(torch.autograd.Function):
    """
    Custom autograd Function implementing the CSS backward pass.

    Forward pass:  identity (no-op).
    Backward pass: scale gradient frequency bands selectively.

    Gradient scaling in Fourier space:
        G_out[band_i] = α_active * G_in[band_i]   if i == active_band
                      = α_silent * G_in[band_i]   otherwise

    This means the optimizer effectively sees a per-frequency learning
    rate, similar to Adam's per-parameter adaptive rates but operating
    in the spectral domain of the hidden representation.

    Why rfft?  The activations are real-valued, so rfft halves the
    frequency dimension without information loss.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor,
                active_band: int, n_bands: int,
                alpha_a: float, alpha_s: float) -> torch.Tensor:
        ctx.save_for_backward(z)
        ctx.active_band = active_band
        ctx.n_bands     = n_bands
        ctx.alpha_a     = alpha_a
        ctx.alpha_s     = alpha_s
        ctx.d           = z.shape[-1]
        return z.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # Transform gradient to frequency domain
        grad_f = torch.fft.rfft(grad, dim=-1)
        L      = grad_f.shape[-1]

        # Partition frequency axis into n_bands equal-width bands
        splits = torch.linspace(0, L, ctx.n_bands + 1).long()

        out = torch.zeros_like(grad_f)
        for i in range(ctx.n_bands):
            alpha = ctx.alpha_a if i == ctx.active_band else ctx.alpha_s
            out[..., splits[i] : splits[i + 1]] = (
                alpha * grad_f[..., splits[i] : splits[i + 1]]
            )

        # Transform back to spatial domain
        grad_spatial = torch.fft.irfft(out, n=ctx.d, dim=-1)
        return grad_spatial, None, None, None, None


class CSSModule(nn.Module):
    """
    Cyclic Spectral Scheduling (CSS) module.

    Wraps _CSSFunction to provide epoch-aware band cycling.  Place this
    as an intermediate layer in the network; it acts as an identity
    function during the forward pass but shapes gradients during back-
    propagation to counteract spectral bias.

    Spectral bias (Rahaman et al., 2019) refers to the empirical
    observation that neural networks learn low-frequency functions much
    faster than high-frequency ones.  CSS addresses this by:

        1. Dividing the gradient spectrum into n_bands bands.
        2. Cycling the "active" band every T epochs.
        3. Attenuating all other bands by α_s (< 1).

    When a band is active, it receives full gradient signal (α_a = 1.0),
    allowing the optimizer to converge those frequencies.  When silent,
    gradients in that band are slowed (α_s < 1), preventing premature
    convergence while the active band is being resolved.

    Args:
        n_bands (int): Number of spectral bands to partition into.
        T       (int): Epochs per band before cycling to the next.
        alpha_a (float): Gradient scale for the active band (default 1.0).
        alpha_s (float): Gradient scale for silent bands (< 1.0).

    Usage::

        css = CSSModule(n_bands=4, T=2, alpha_a=1.0, alpha_s=0.4)
        # inside training loop:
        css.update_epoch(epoch)
        z_out = css(z)          # forward = identity; backward = spectral gate
    """

    def __init__(self,
                 n_bands: int = 4,
                 T: int = 2,
                 alpha_a: float = 1.0,
                 alpha_s: float = 0.4):
        super().__init__()
        self.n_bands = n_bands
        self.T       = T
        self.alpha_a = alpha_a
        self.alpha_s = alpha_s
        self.epoch   = 0

    def update_epoch(self, e: int):
        """Call at the start of each epoch to advance the band schedule."""
        self.epoch = e

    @property
    def active_band(self) -> int:
        """Which frequency band is currently active."""
        return (self.epoch // self.T) % self.n_bands

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return _CSSFunction.apply(
            z, self.active_band, self.n_bands, self.alpha_a, self.alpha_s
        )


# ─────────────────────────────────────────────────────────────
#  Per-Feature Branch Network
# ─────────────────────────────────────────────────────────────

class BranchNetwork(nn.Module):
    """
    Single-feature encoder with PMA nonlinearity.

    Each scalar input feature x_i ∈ ℝ is projected to a d_h-dimensional
    embedding via a linear layer, then passed through PMA to produce a
    (2/3)*d_h vector, which is projected back to d_h.

    Architecture::

        x_i (scalar) → Linear(1, d_h) → PMA → Linear(pma.out_dim, d_h) → h_i

    Why per-feature branches?
        - Domain physics couples features non-linearly (e.g., Weibull
          shape × critical length × fibre diameter in the Kelly–Tyson
          model).  A shared encoder loses these single-feature semantics.
        - Separate branches allow each feature's representation to be
          fine-tuned independently via the subsequent MHA cross-attention.
        - The total parameter count remains manageable because each branch
          is shallow (2 linears + PMA).

    Args:
        d_h (int): Hidden dimension (must be divisible by 3 for PMA).
    """

    def __init__(self, d_h: int = 96):
        super().__init__()
        assert d_h % 3 == 0, f"BranchNetwork requires d_h divisible by 3, got {d_h}"
        self.fc1 = nn.Linear(1, d_h)
        self.pma = PMA(d_h)
        self.fc2 = nn.Linear(self.pma.out_dim, d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, 1) — a single normalised feature.
        Returns:
            Tensor of shape (B, d_h).
        """
        return self.fc2(self.pma(self.fc1(x)))


# ─────────────────────────────────────────────────────────────
#  Main CFRP-PINN
# ─────────────────────────────────────────────────────────────

class CFRP_PINN(nn.Module):
    """
    Physics-Informed Neural Network for CFRP Micromechanical Modelling.

    Complete architecture:

        Input (B, 18) → 18 BranchNetworks (B, 18, d_h)
                       → Multi-Head Attention  (B, 18, d_h)
                       → Flatten + Linear Projection (B, d_h)
                       → CSS (identity fwd, spectral grad gate bwd)
                       → PMA (B, 2/3*d_h)
                       → Linear → (B, 2)   [σ_HT, E_HT]

    The 18 input features are:
        [0]  w_f    — fibre volume fraction
        [1]  weib_a — Weibull scale parameter (fibre strength distribution)
        [2]  weib_b — Weibull shape parameter
        [3]  L_min  — minimum fibre length
        [4]  L_max  — maximum fibre length
        [5]  L_crit — critical fibre transfer length
        [6]  D      — fibre diameter
        [7-12]      — double-Gaussian orientation distribution params
                      (α₁,β₁,γ₁, α₂,β₂,γ₂)
        [13] E_m    — matrix Young's modulus
        [14] σ_m   — matrix tensile strength
        [15] E_f    — fibre Young's modulus
        [16] σ_f   — fibre tensile strength
        [17] ν     — Poisson's ratio (Halpin-Tsai)

    Args:
        d_h     (int): Hidden dimension per branch (divisible by 3).
        n_heads (int): Number of attention heads (must divide d_h).
        n_bands (int): Spectral bands for CSS.
        n_feat  (int): Number of input features (default 18).
        css_T   (int): Epochs per CSS band cycle.
        css_alpha_s (float): Silent-band gradient attenuation.

    Example::

        model = CFRP_PINN(d_h=96, n_heads=4, n_bands=4)
        x     = torch.randn(64, 18)   # normalised features
        out   = model(x)              # (64, 2) → [σ̂_HT, Ê_HT] normalised
    """

    def __init__(self,
                 d_h: int = 96,
                 n_heads: int = 4,
                 n_bands: int = 4,
                 n_feat: int = 18,
                 css_T: int = 2,
                 css_alpha_s: float = 0.4):
        super().__init__()
        assert d_h % 3 == 0, f"d_h must be divisible by 3, got {d_h}"
        assert d_h % n_heads == 0, f"d_h must be divisible by n_heads, got {d_h}/{n_heads}"

        self.n_feat   = n_feat
        self.d_h      = d_h

        # ── Per-feature encoders ──────────────────────────────
        self.branches = nn.ModuleList(
            [BranchNetwork(d_h) for _ in range(n_feat)]
        )

        # ── Cross-feature attention ───────────────────────────
        # Operates on sequence of length n_feat with dim d_h
        self.mha = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_h)

        # ── Projection from concatenated branches ─────────────
        self.proj = nn.Linear(n_feat * d_h, d_h)
        self.proj_norm = nn.LayerNorm(d_h)

        # ── Spectral gradient scheduling ──────────────────────
        self.css = CSSModule(n_bands=n_bands, T=css_T, alpha_s=css_alpha_s)

        # ── Output head with PMA nonlinearity ─────────────────
        self.pma_out = PMA(d_h)
        self.out     = nn.Linear(self.pma_out.out_dim, 2)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for all linear layers; identity-friendly for PMA paths."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def update_epoch(self, epoch: int):
        """Propagate epoch counter into the CSS module."""
        self.css.update_epoch(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Normalised input tensor of shape (B, n_feat).
        Returns:
            Normalised predictions (B, 2) → columns are [σ̂_HT, Ê_HT].
        """
        B = x.size(0)

        # ── Branch encoding: one scalar per branch ────────────
        # Stack to (B, n_feat, d_h) for MHA
        H = torch.stack(
            [branch(x[:, i : i + 1]) for i, branch in enumerate(self.branches)],
            dim=1
        )                                               # (B, n_feat, d_h)

        # ── Cross-feature self-attention with residual ────────
        H_attn, _ = self.mha(H, H, H)
        H = self.attn_norm(H + H_attn)                 # (B, n_feat, d_h)

        # ── Flatten → project → normalise ────────────────────
        z = self.proj_norm(
            self.proj(H.reshape(B, -1))
        )                                               # (B, d_h)

        # ── Spectral gradient gate (forward = identity) ───────
        z = self.css(z)                                 # (B, d_h)

        # ── Output head ───────────────────────────────────────
        return self.out(self.pma_out(z))                # (B, 2)
