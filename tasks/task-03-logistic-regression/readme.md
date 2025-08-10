# Logistic Regression from Scratch

This exercise guides you through implementing a **logistic regression** classifier using only NumPy.

## Objective

You will complete the training function of a simple neural unit (`LogisticNeuron`) that learns to separate data into two classes using gradient descent.

---

## Logistic Regression Theory

Logistic regression is used to model the probability that a given input belongs to a particular class.

### 1. **Model Equation**

Given an input vector $\mathbf{x} \in \mathbb{R}^n$, the logistic neuron computes:

$$
z = \mathbf{w}^T \mathbf{x} + b
$$

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:
- $\mathbf{w}$ is the weights vector,
- $b$ is the bias,
- $\sigma(z)$ is the sigmoid function,
- $\hat{y}$ is the predicted probability.

---

### 2. **Loss Function (Binary Cross-Entropy)**

To evaluate how well the model is performing, we use the **binary cross-entropy loss**:

$$
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)} + \epsilon) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)} + \epsilon) \right]
$$

Where:
- $m$ is the number of training examples,
- $y^{(i)}$ is the true label for example $i$,
- $\hat{y}^{(i)}$ is the predicted probability for example $i$,
- $\epsilon$ is a small constant (e.g., $1e^{-8}$) to avoid $\log(0)$.

---

### 3. **Gradient Descent Update Rules**

To minimize the loss, you perform gradient descent with the following update rules:

**Weight Gradient**:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)}
$$

**Bias Gradient**:
$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

**Parameter Update**:
```python
weights -= learning_rate * grad_w
bias -= learning_rate * grad_b
