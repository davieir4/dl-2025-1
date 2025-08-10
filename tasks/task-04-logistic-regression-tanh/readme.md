# Logistic Neuron with Tanh Activation and MSE Loss

This task is an extension of the basic logistic regression model. Instead of using the traditional sigmoid activation with binary cross-entropy loss, you'll implement:

- **Hyperbolic tangent (tanh)** as the activation function.
- **Mean Squared Error (MSE)** as the loss function.
- Labels transformed from `{0, 1}` to `{-1, +1}`.

---

## 🎯 Task Objective

Update the `LogisticNeuron` class to:

1. Use the **tanh** function for output activation.
2. Replace the binary cross-entropy loss with **Mean Squared Error**.
3. Transform the labels to match the output range of `tanh`.

---

## 🧠 Theoretical Background

### 🔁 Activation Function: Tanh

The `tanh` function is defined as:

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

It outputs values in the range **(-1, 1)** and is zero-centered.

---

### 🏷️ Label Transformation

Since `tanh(z)` produces values between -1 and 1, we must encode the target labels accordingly:

```python
y_tanh = 2 * y - 1
