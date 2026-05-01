# 🚀 CFRP-PINN

**Physics-Informed Neural Network for Composite Materials**

A novel deep learning framework for **Carbon Fiber Reinforced Polymer (CFRP)** micromechanical modeling, integrating **first-principles physics** with **next-generation neural architectures**.

---

## 🌟 Key Novelty

This project introduces **three original contributions** that go beyond standard PINNs:

### 1. 🧠 Polynomial Multiplicative Attention (PMA)

* Activation-free nonlinearity
* Enables **implicit quadratic feature interactions**
* Avoids vanishing gradients and dead neurons
* Replaces ReLU/GELU entirely

### 2. 🌊 Cyclic Spectral Scheduling (CSS)

* Operates in **Fourier space during backpropagation**
* Cyclically focuses on frequency bands
* Mitigates **spectral bias** in neural networks
* Improves convergence on both low & high-frequency components

### 3. 🌿 Branch-wise Feature Encoding + Attention Fusion

* Each input feature gets its **own neural encoder**
* Cross-feature relationships learned via **Multi-Head Attention**
* Preserves **physical semantics of individual variables**

---

## 🧪 Physics Integration

Unlike purely data-driven models, this network embeds **micromechanical theory directly into training**.

### Predicted Properties

* **Ultimate Tensile Strength (σ_HT)**
* **Young’s Modulus (E_HT)**

### Governing Physics

* Modified Rule of Mixtures (Kelly–Tyson)
* Halpin–Tsai Model
* Weibull Fiber Length Distribution
* Orientation Distribution via Double Gaussian

> 📌 *Physics residual implementation: add your file link here*

---

## ⚙️ Architecture Overview

```
Input (18 features)
   ↓
[18 × Branch Networks]
   ↓
Multi-Head Attention
   ↓
Projection Layer
   ↓
Cyclic Spectral Scheduling (CSS)
   ↓
PMA Head
   ↓
Output → [σ_HT, E_HT]
```

> 📌 *Core model implementation: add your file link here*

---

## 📊 Loss Function

Total loss combines **data fidelity + physics constraints**:

[
\mathcal{L} = \mathcal{L}*{data} + \lambda(t) \cdot \mathcal{L}*{physics}
]

### Progressive Physics Weighting

[
\lambda(t) = \lambda_{max} \cdot \left(1 - e^{-t/\tau}\right)
]

* Prevents early training instability
* Gradually enforces physical consistency

---

## 🧩 Input Features (18)

| Category    | Features                           |
| ----------- | ---------------------------------- |
| Fiber       | volume fraction, diameter          |
| Weibull     | scale (a), shape (b), L_min, L_max |
| Mechanics   | critical length, moduli, strengths |
| Orientation | double Gaussian parameters         |
| Matrix      | modulus, strength                  |
| Other       | Poisson’s ratio                    |

---

## 🔬 Why This Matters

### Traditional ML Models

* ❌ Ignore physics → poor extrapolation
* ❌ Require large datasets

### CFRP-PINN

* ✅ Enforces **physical laws explicitly**
* ✅ Works with **limited data**
* ✅ Produces **physically consistent predictions**
