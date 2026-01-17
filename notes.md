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


## Day 04 — Logistic Regression & The Sigmoid Function

### The Necessity of Sigmoid
Linear models are unbounded and unsuitable for probabilistic classification.

Classification requires a mapping into the interval:

$$
(0, 1)
$$

The sigmoid function provides a smooth, monotonic transformation that enables
interpretable probabilities:

$$
P(y=1 \mid X) = \sigma(z)
$$

---

### Logits and Evidence
The linear score:

$$
z = Xw + b
$$

is referred to as the *logit*.  
Geometrically, $$z$$ represents accumulated evidence from the input features,
while the sigmoid converts this evidence into confidence.

---

### The Decision Boundary
Classification occurs where predicted probability equals 0.5:

$$
P(y=1 \mid X) = 0.5
$$

This condition is satisfied exactly when:

$$
z = 0
$$

Thus, the decision boundary is a linear hyperplane in feature space.

---

### Vanishing Gradients
The derivative of the sigmoid function is:

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

As $$|z|$$ becomes large, $$\sigma'(z)$$ approaches zero.

This causes gradients to vanish for samples the model is already highly confident
about, reducing their contribution to learning and slowing convergence.


## Day 05 — Log Loss & Numerical Stability

### Why MSE Fails
Mean Squared Error treats prediction errors linearly and produces non-convex cost
surfaces when paired with a sigmoid activation.

This leads to weak gradients and unreliable convergence in classification tasks.

Log Loss preserves convexity for logistic regression, guaranteeing a single global
minimum.

---

### Epsilon Clipping
Because the logarithm is undefined at zero and one:

$$
\log(0) \;\; \text{is undefined}
$$

predicted probabilities must be clipped to:

$$
[\epsilon, 1 - \epsilon]
$$

with a typical choice:

$$
\epsilon = 10^{-15}
$$

This prevents numerical instability during both loss and gradient computation.

---

### Gradient Intuition
The gradient of Log Loss with respect to predictions scales inversely with the
model’s confidence error.

As a result:
- mildly wrong predictions contribute small gradients
- **confidently wrong predictions dominate learning**

This ensures steepest descent occurs where the model is most incorrect.

---

### Analytical vs. Numerical Verification
Analytical derivatives must always be validated against numerical gradients using
finite-difference approximations.

Agreement between the two confirms:
- correct implementation
- stable learning dynamics
- readiness for integration into training loops


## Day 06 — Logistic Regression Training Dynamics

### Objective Optimization
Training adjusts parameters $$w$$ and $$b$$ to align predicted probabilities with
actual labels.

Geometrically, this corresponds to moving the decision boundary to minimize
confidence error across the dataset.

---

### The Role of Log Loss
Log Loss serves as the continuous driver of learning.

Even after classification accuracy stabilizes, loss continues to decrease as the
model increases certainty in correct predictions and reduces overconfidence in
incorrect ones.

---

### Geometry of the Boundary
The decision boundary is defined where:

$$
P(y = 1 \mid X) = 0.5
$$

which occurs exactly when:

$$
z = Xw + b = 0
$$

Training is equivalent to rotating and translating this hyperplane to maximize
class separation in feature space.

---

### Convergence Indicators
Stable convergence is characterized by:
- a flattening loss curve
- diminishing gradient magnitudes
- stable parameter updates

Because logistic regression with log loss is convex, convergence implies the
optimizer has reached the global minimum of the cost surface.
