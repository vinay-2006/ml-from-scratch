# ML From Scratch

This repository implements core machine learning algorithms from first principles
using **NumPy only**.

The objective is to translate mathematical theory into **functional, verifiable
systems**, with emphasis on:
- geometric intuition
- numerical correctness
- vectorized computation
- engineering tradeoffs

High-level machine learning libraries are intentionally avoided.

---

## 🟦 Week B — ML From Scratch (Implementation Phase)

This phase transitions from mathematical foundations to executable learning systems.
Each day builds reusable primitives that are later composed into full training loops.

The focus is not speed, but **correctness, clarity, and system behavior**.

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
- underfitting due to poor projection alignment
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
**orthogonal projection of the target vector $$y$$ onto the column space of $$X$$**.

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
- Matrix inversion scales as $$O(d^3)$$
- The solution requires $$X^T X$$ to be invertible
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

## Day 03 — Linear Regression: Batch Gradient Descent

### Iterative Optimization
Implemented a scalable batch gradient descent training loop to minimize the
Mean Squared Error (MSE) cost function iteratively.

This replaces closed-form matrix inversion with incremental parameter updates.

---

### Vectorized Calculus
Parameter updates follow:

$$
w = w - \eta \frac{\partial J}{\partial w}, \quad
b = b - \eta \frac{\partial J}{\partial b}
$$

Gradients are computed using matrix–vector products, enabling fully vectorized
multi-parameter updates without explicit loops.

---

### Convergence Engineering
Integrated convergence controls into the training loop:
- Early stopping based on cost improvement threshold
- Maximum iteration safeguards
- Empirical analysis of learning rate stability and divergence

These mechanisms ensure numerically stable optimization.

---

### Reusable Component
Encapsulated gradient descent logic into a reusable `LinearRegressionGD` class,
establishing the first production-style model in the *ML From Scratch* library.

Day 03 completes the transition from analytical solutions to scalable learning systems.

## Day 04 — Logistic Regression: The Sigmoid Function

### Probabilistic Transformation
Linear regression produces unbounded outputs and fails for classification tasks.
To address this, implemented the Sigmoid activation function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

which transforms the linear score $$z = Xw + b$$ into a bounded probability
$$P(y=1 \mid X) \in (0, 1)$$.

---

### Decision Boundary Geometry
Classification is interpreted geometrically as a spatial partitioning problem.

The decision boundary occurs where:

$$
z = Xw + b = 0
$$

This linear hyperplane divides the feature space into two regions corresponding
to predicted class probabilities above and below 0.5.

---

### Numerical Stability Engineering
Implemented input-range limiting for the sigmoid computation to prevent numerical
overflow in the exponential function.

This ensures stable forward passes even for large-magnitude logits and is a
necessary safeguard for production-grade implementations.

---

### Gradient Behavior Analysis
Analyzed the derivative of the sigmoid function:

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

This reveals the vanishing gradient risk in regions of extreme model confidence,
where updates become small and learning slows.

Day 04 establishes the probabilistic foundation required for logistic regression
optimization and classification loss functions.


## Day 05 — Logistic Regression: Log Loss & Derivatives

### Objective Function Analysis
Evaluated the failure of Mean Squared Error (MSE) for classification tasks and
implemented Binary Cross-Entropy (Log Loss) as the correct probabilistic objective
for logistic regression.

Log Loss evaluates the correctness of **model confidence**, not just numerical
distance between predictions and labels.

---

### Numerical Stability Primitives
Engineered epsilon-clipping logic to prevent logarithmic singularities when computing:

$$
\log(0) \quad \text{and} \quad \log(1)
$$

Predicted probabilities are constrained to the interval:

$$
[\epsilon, 1 - \epsilon]
$$

ensuring numerically stable loss and gradient computation.

---

### Learning Signal Verification
Developed a numerical gradient checker using finite differences to validate the
analytical derivatives of the Log Loss function.

Close agreement between analytical and numerical gradients confirms correctness
and reliability of the learning signal.

---

### Confidence-Based Punishment
Visualized Log Loss as a function of predicted probability to demonstrate how
confident incorrect predictions are penalized **exponentially**, forcing the model
to prioritize high-error samples during optimization.

Day 05 establishes the correct objective function required for stable and
interpretable classification learning.


## Day 06 — Logistic Regression: Gradient Descent & Decision Boundaries

### End-to-End Synthesis
Integrated linear scoring, sigmoid activation, and log loss into a complete binary
classification training pipeline.

The model computes:
- a linear score $$z = Xw + b$$
- probabilistic predictions $$\hat{y} = \sigma(z)$$
- optimization via Binary Cross-Entropy (Log Loss)

This establishes logistic regression as a fully trainable system.

---

### Decision Boundary Optimization
Visualized the geometric evolution of the decision boundary defined by:

$$
z = Xw + b = 0
$$

Training progressively rotates and translates this hyperplane to partition feature
space into probabilistic regions corresponding to class labels.

---

### Gradient Chain Rule Implementation
Leveraged the simplified gradient:

$$
\nabla_w J = \frac{1}{n} X^T (\hat{y} - y)
$$

This cancellation of sigmoid and log loss derivatives yields numerically stable and
efficient parameter updates, closely resembling linear regression gradients.

---

### Loss vs. Accuracy Analysis
Analyzed training dynamics to distinguish between:
- **Log Loss** — a continuous optimization objective
- **Accuracy** — a discrete evaluation metric

Observed that loss continues to decrease even when accuracy plateaus, reflecting
increasing model confidence rather than class count changes.

Day 06 completes the end-to-end training loop for logistic regression.
