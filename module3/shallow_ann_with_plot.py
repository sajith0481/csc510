
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(1, output_size)
        self.losses = []

    def feedforward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def backpropagate(self, X, y, learning_rate):
        output_error = y - self.output
        d_output = output_error * sigmoid_derivative(self.output)

        hidden_error = d_output.dot(self.weights_hidden_output.T)
        d_hidden = hidden_error * sigmoid_derivative(self.hidden)

        self.weights_hidden_output += self.hidden.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagate(X, y, learning_rate)
            loss = mse_loss(y, self.output)
            self.losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()
        plt.savefig("training_loss.png")
        print("Training loss plot saved as 'training_loss.png'")


# Sample training data
X = np.array([[i, i+1] for i in range(1, 6)])
y = np.array([[i+2] for i in range(1, 6)])

# Normalize
X_max = np.max(X)
y_max = np.max(y)
X = X / X_max
y = y / y_max

# Train the ANN
ann = SimpleANN(input_size=2, hidden_size=4, output_size=1)
ann.train(X, y, epochs=1000, learning_rate=0.1)
ann.plot_loss()

# User input for testing
try:
    user_input = input("Enter two numbers separated by a comma (e.g., 6,7): ")
    test_input = np.array([[float(i) for i in user_input.split(",")]])
    normalized_input = test_input / X_max
    predicted_output = ann.feedforward(normalized_input)
    result = predicted_output * y_max
    print(f"Predicted next number in the sequence: {result[0][0]:.2f}")
except:
    print("Invalid input. Please enter two numbers separated by a comma.")
