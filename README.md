# ML From Scratch

This repository implements core machine learning algorithms from first principles using NumPy.
The objective is to translate mathematical theory into functional, verifiable systems without relying on high-level machine learning libraries.

---

## 🟦 Week B — ML From Scratch (Implementation Phase)

This phase transitions from mathematical theory to functional systems by implementing core algorithms using only NumPy.  
The goal is to establish high-fidelity models by explicitly deriving and coding every primitive, with emphasis on geometric intuition, numerical correctness, and vectorized computation.

---

### Day 01 — Linear Regression: Geometric Projection & Residuals

#### Architectural Framework
Implemented the forward prediction engine:

$$
\hat{y} = Xw + b
$$

Linear regression is treated as a geometric projection of feature vectors into target space rather than as an optimization problem.

---

#### Residual Vector Analysis
Residuals are defined as:

$$
r = y - \hat{y}
$$

Residuals are visualized as vertical geometric distances between ground truth values and model predictions.  
This establishes the conceptual basis for later cost minimization and optimization.

---

#### Parameter Sensitivity
Manual parameter sweeps over slope and bias were conducted to observe:

- Underfitting due to poor projection alignment
- Systematic residual patterns
- Improved geometric alignment with better parameter choices

This isolates the effect of parameters before introducing automated optimization.

---

#### Numerical Standards
- Enforced strict shape safety across all matrix and vector operations
- Relied on NumPy broadcasting to ensure correct and efficient vectorized computation
- Avoided implicit reshaping or silent dimension expansion

---

Day 01 formalizes *how predictions are generated* in linear regression.  
Cost functions, gradients, and optimization are intentionally deferred to later days.
