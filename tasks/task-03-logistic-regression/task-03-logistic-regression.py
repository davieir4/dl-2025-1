import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LogisticNeuron:
    def __init__(self, input_dim, learning_rate=0.1, epochs=1000):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []
    
    def sigmoid(self, z):
        ### START CODE HERE ###
        s = 1 / (1 + np.exp(-z))
        ### END CODE HERE ###
        return s
    
    def predict_proba(self, X):
        ### START CODE HERE ###
        z = np.dot(X, self.weights) + self.bias
        a = self.sigmoid(z)
        ### END CODE HERE ###
        return a
    
    def predict(self, X):
        prediction = (self.predict_proba(X) >= 0.5).astype(int)
        return prediction
    
    def train(self, X, y):
        for _ in range(self.epochs):
            ### START CODE HERE ###
            # Forward pass
            y_pred = self.predict_proba(X)

            # Compute error
            error = y_pred - y

            # Gradients
            grad_w = np.dot(X.T, error) / len(y)
            grad_b = np.sum(error) / len(y)

            # Update weights and bias
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
            self.loss_history.append(loss)
            ### END CODE HERE ###

def generate_dataset():
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.0)
    return X, y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Logistic Regression Output')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

def plot_loss(model):
    plt.plot(model.loss_history, 'k.')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over Training Iterations')
    plt.show()

def main():
    # Generate dataset
    X, y = generate_dataset()

    # Train the model
    neuron = LogisticNeuron(input_dim=2, learning_rate=0.1, epochs=100)
    neuron.train(X, y)

    # Plot decision boundary
    plot_decision_boundary(neuron, X, y)

    # Plot loss over training iterations
    plot_loss(neuron)

if __name__ == "__main__":
    main()
