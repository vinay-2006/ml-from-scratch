# ML From Scratch — Technical Notes

---

## Day 01 — Linear Regression Geometry & Forward Pass

### Prediction Logic
Linear regression defines a mapping:

$$
f: \mathbb{R}^d \rightarrow \mathbb{R}
$$

Predictions are computed as:

$$
\hat{y} = Xw + b
$$

where the weight vector $$w$$ defines the direction of projection in feature space.

---

### Geometric Residuals
Residuals are defined as:

$$
r = y - \hat{y}
$$

They represent **vertical error vectors** between ground truth values and model
predictions.

Learning later seeks to minimize the aggregate magnitude of these vectors.

---

### Role of Parameters
- $$w$$ controls the **orientation** of the prediction line or hyperplane
- $$b$$ acts as a **translation parameter**, shifting predictions vertically

Small changes in $$w$$ significantly affect projection alignment.

---

### Broadcasting Primitives
The implementation relies on NumPy broadcasting to compute:

$$
\hat{y} = Xw + b
$$

where the scalar bias term $$b$$ is automatically expanded across the output vector.
This pattern is critical for high-performance vectorized ML code.

---

### Transition to Cost
Day 01 establishes *how predictions are generated*.

Residual vectors are observable, but no single scalar metric exists to evaluate
global model performance. This motivates the introduction of cost functions.

---

## Day 02 — The Normal Equation & Closed-Form Solutions

### Least Squares as Projection
The normal equation computes the **orthogonal projection** of the target vector $$y$$
onto the column space of $$X$$.

The resulting projection corresponds to the predictions $$\hat{y} = Xw$$.

---

### Augmentation Strategy
Bias handling is unified by augmenting the data matrix:

$$
X \in \mathbb{R}^{n \times d}
\;\;\longrightarrow\;\;
[\mathbf{1}, X] \in \mathbb{R}^{n \times (d+1)}
$$

with an augmented weight vector:

$$
\begin{bmatrix}
b \\
w
\end{bmatrix}
$$

This allows all parameters to be learned via a single dot product.

---

### Numerical Standards
- $$X \in \mathbb{R}^{n \times d}$$ — feature matrix
- $$w \in \mathbb{R}^{d}$$ — weight vector
- $$y \in \mathbb{R}^{n}$$ — target vector

Shape correctness is treated as a first-class constraint.

---

### The Inversion Bottleneck
The normal equation requires computing:

$$
(X^T X)^{-1}
$$

This operation:
- scales as $$O(d^3)$$
- becomes impractical for large feature counts
- fails if features are linearly dependent

This motivates regularization or iterative methods.

---

### Why Closed-Form Solutions Matter
Despite scalability limits, the normal equation is valuable because it:
- provides geometric clarity
- exposes numerical failure modes
- establishes a baseline for optimization methods

---

### Transition to Gradient Descent
The limitations of matrix inversion justify the move to **iterative optimization**.

Gradient Descent replaces exact inversion with incremental parameter updates,
scaling to large datasets while using the same prediction and residual primitives.

---

## Day 03 — Batch Gradient Descent & Cost Surfaces

### The Cost Function (MSE)
While residuals provide local error signals, Mean Squared Error (MSE) aggregates
them into a global scalar objective:

$$
J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

For linear regression, this cost surface forms a convex bowl, guaranteeing a single
global minimum.

---

### Gradient Logic
Gradients represent the direction of steepest ascent of the cost surface.

By updating parameters in the opposite direction of the gradient, optimization
proceeds “downhill” toward the minimum:

$$
w \leftarrow w - \eta \nabla_w J, \quad
b \leftarrow b - \eta \nabla_b J
$$

This replaces exact algebraic solutions with controlled numerical descent.

---

### Hyperparameter Sensitivity
The learning rate $$\eta$$ is the most critical hyperparameter:
- too small → slow convergence
- too large → overshooting and divergence

Observed instability directly reflects the curvature of the cost surface.

---

### Engineering Perspective
Gradient Descent trades mathematical exactness for scalability.

The same prediction and residual primitives from Days 01 and 02 are reused,
demonstrating that optimization is an extension of the existing system—not a
new model.
