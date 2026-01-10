# ML From Scratch

This repository implements core machine learning algorithms from first principles
using **NumPy only**.

The objective is to translate mathematical theory into **functional, verifiable
systems**, with emphasis on:
- geometric intuition
- numerical correctness
- vectorized computation
- engineering tradeoffs

High-level ML libraries are intentionally avoided.

---

## 🟦 Week B — ML From Scratch (Implementation Phase)

This phase transitions from mathematical foundations to executable learning systems.
Each day builds reusable primitives that are later composed into full training loops.

The focus is not speed, but **correctness and understanding**.

---

## Day 01 — Linear Regression: Geometric Projection & Residuals

### Forward Prediction
Implemented the linear forward pass:

$$
\hat{y} = Xw + b
$$

Linear regression is treated as a **geometric projection problem**, not an
optimization problem.

---

### Residual Geometry
Residuals are defined as:

$$
r = y - \hat{y}
$$

They are interpreted as **vertical geometric distances** between the prediction
line and observed data points.

Manual parameter sweeps were used to observe:
- underfitting due to poor alignment
- systematic residual patterns
- improved geometric alignment with better parameter choices

This isolates model behavior before introducing optimization.

---

### Numerical Standards
- Enforced strict shape safety across all matrix operations
- Relied on NumPy broadcasting for efficient vectorized computation
- Avoided implicit reshaping or silent dimension expansion

Day 01 formalizes *how predictions are generated*.

---

## Day 02 — Linear Regression: The Normal Equation

### Analytical Optimization
Implemented the closed-form least squares solution:

$$
w = (X^T X)^{-1} X^T y
$$

This computes the **exact global minimum** of the squared error objective.

---

### Least Squares as Projection
The normal equation is interpreted geometrically as computing the
**orthogonal projection of the target vector $y$ onto the column space of $X$**.

This framing explains why the solution exists and when it fails.

---

### Matrix Augmentation
The bias term was absorbed into the model by augmenting the data matrix:

$$
\hat{y} = [\mathbf{1}, X]
\begin{bmatrix}
b \\
w
\end{bmatrix}
$$

This allows all parameters to be learned through a single dot product.

---

### Stability & Scalability Analysis
- Matrix inversion scales as $O(d^3)$
- The solution requires $X^T X$ to be invertible
- Linearly dependent features cause failure (singularity)

From an engineering perspective, explicit inversion is avoided in favor of
numerical solvers, but the inverse is shown here for mathematical clarity.

---

### Transition to Iterative Optimization
While the normal equation is exact, it does **not scale** to large models.

This inversion bottleneck motivates the transition to **iterative optimization**
(Gradient Descent), where parameters are learned progressively using the same
prediction and residual primitives developed in Days 01 and 02.

---

## Next Step — Day 03
Implementation of **Gradient Descent** from scratch, introducing:
- cost functions
- gradients
- training loops
- convergence behavior
