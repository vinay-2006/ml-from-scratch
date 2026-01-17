# 📑 Week B — Engineering Notes

Lessons learned while building, breaking, and fixing ML systems from scratch.
These notes focus on *failure modes* and *why fixes worked*.

---

## 01 | Geometry Reframed Learning

Early on, residuals were treated as abstract error terms.

Plotting them as **projection distances** revealed that residuals *define* the learning signal. Once this was visualized, gradient descent behavior stopped feeling arbitrary.

📎 See: Day 01 notebook — residual plots.

---

## 02 | Exact Solutions Fail Quietly

The Normal Equation worked — until it didn’t.

Observed issues:
- singular matrices with correlated features
- exploding weights for poorly scaled inputs

Lesson:
> Mathematical exactness does not imply numerical robustness.

---

## 03 | Optimization Was the Real Bottleneck

Initial Gradient Descent runs diverged.

Root causes:
- learning rate > 0.1 on curved cost surfaces
- ignoring gradient magnitude diagnostics

Fix:
- learning rate sweeps
- early stopping based on loss delta

📎 See: Day 03 loss divergence vs convergence plots.

---

## 04 | Probability Introduced New Failure Modes

Switching to sigmoid introduced:
- saturation for `|z| > 10`
- vanishing gradients
- overflow for `z > ~60`

Sigmoid clipping was added **after observing `inf` values** in forward passes.

📎 See: Day 04 sigmoid saturation plots.

---

## 05 | Loss Design Changed Everything

Using MSE with sigmoid caused:
- weak gradients for confident mistakes
- non-convex behavior

Switching to Log Loss:
- restored convexity
- amplified gradients for confident errors
- simplified gradient expressions

📎 See: Day 05 log-loss vs MSE comparison.

---

## 06 | Loss ≠ Accuracy

Accuracy plateaued early.
Log Loss kept decreasing.

This showed the model was learning *confidence calibration*, not just labels.

Key realization:
> Accuracy alone hides learning progress.

---

## 07 | Metrics Encode Cost

On the Iris dataset:
- accuracy remained high
- recall varied significantly with threshold

Precision / Recall revealed trade-offs accuracy could not.

Final takeaway:
> Metrics are business decisions disguised as math.
