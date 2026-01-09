## Day 01 — Linear Regression Geometry & Forward Pass

### Prediction Logic
Linear regression defines a mapping:

$$
f: \mathbb{R}^d \rightarrow \mathbb{R}
$$

where the input feature matrix $$X$$ is transformed by a weight vector $$w$$ representing the direction of best fit in feature space.

---

### Geometric Residuals
Residuals are defined as:

$$
r = y - \hat{y}
$$

These residuals represent vertical error vectors between ground truth values and model predictions.  
From a geometric perspective, learning seeks to minimize the aggregate magnitude of these residual vectors.

---

### Role of Weight ($$w$$)
The weight vector controls the orientation (rotation) of the prediction line or hyperplane.

Small changes in $$w$$ significantly affect:
- The direction of projection
- The alignment between predicted outputs and the target distribution

---

### Role of Bias ($$b$$)
The bias term acts as a translation parameter.

It shifts the prediction function vertically, allowing the model to fit data distributions that do not pass through the origin.

---

### Broadcasting Primitives
The implementation relies on NumPy broadcasting to compute:

$$
\hat{y} = Xw + b
$$

where the scalar bias term $$b$$ is automatically expanded across the output vector.  
This pattern is critical for writing high-performance, vectorized machine learning code.

---

### Transition to Cost
Day 01 establishes *how predictions are generated*.

While residual vectors are observable, there is no single scalar metric to evaluate global model performance.  
This motivates the introduction of a cost function to aggregate residual behavior, which is addressed in subsequent days.
