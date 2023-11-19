import numpy as np

# Class for activation functions
class ActivationFunctions:
    def __init__(self, activation_type="sigmoid"):
        self.set_activation(activation_type)

    def set_activation(self, activation_type):
        # Set activation functions based on the specified type
        if activation_type == "sigmoid":
            self.activation_function = lambda x: 1 / (1 + np.exp(-x))
            self.derivative_function = lambda x: self.activation_function(x) * (1 - self.activation_function(x))
        elif activation_type == "relu":
            self.activation_function = lambda x: np.maximum(0, x)
            self.derivative_function = lambda x: np.where(x > 0, 1, 0)
        elif activation_type == "linear":
            self.activation_function = lambda x: x
            self.derivative_function = lambda x: np.ones_like(x)
        elif activation_type == "tanh":
            self.activation_function = lambda x: np.tanh(x)
            self.derivative_function = lambda x: 1 - np.tanh(x)**2
        else:
            raise ValueError("Unsupported activation function type")

    def activate(self, x):
        return self.activation_function(x)

    def derivative(self, x):
        return self.derivative_function(x)


# Neural Network class
class MyNeuralNetwork:
    def __init__(self, layers, lr=0.1, momentum=0.1, epochs=1000, fact="sigmoid", valid=0.2):
        # Initialize neural network parameters
        self.L = len(layers)
        self.n = layers.copy()
        self.activation_function = ActivationFunctions(fact)
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.valid = valid

        # Lists to store errors during training and validation
        self.error_training = []
        self.error_validation = []

        # Arrays to store intermediate values during forward and backward passes
        self.h = [np.zeros(size) for size in layers]  # activation function
        self.xi = [np.zeros(size) for size in layers]  # node values

        # Arrays to store weights, biases, errors, and their updates
        self.w = [np.zeros((1, 1))]  # edge weights
        self.theta = [0]  # bias
        self.delta = [np.zeros(size) for size in layers]  # error

        self.d_w = [np.zeros((1, 1))]
        self.d_w_prev = [np.zeros((1, 1))]  # delta weights
        self.d_theta = [np.zeros(size) for size in layers]
        self.d_theta_prev = [np.zeros(size) for size in layers]

        # Initialize weights, biases, and errors arrays based on the specified layers
        for lay in range(1, self.L):
            self.w.append(np.zeros((layers[lay], layers[lay - 1])))
            self.theta.append(np.zeros(layers[lay]))
            self.delta.append(np.zeros(layers[lay]))
            self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))
            self.d_theta.append(np.zeros(layers[lay]))
            self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))
            self.d_theta_prev.append(np.zeros(layers[lay]))

    def __forward(self):
        # Perform forward pass through the network
        for lay in range(1, self.L):
            self.h[lay] = np.dot(self.w[lay], self.xi[lay - 1]) - self.theta[lay]
            self.xi[lay] = self.activation_function.activate(self.h[lay])

    def __backward(self, y):
        # Perform backward pass through the network
        self.delta[self.L - 1] = self.activation_function.derivative(self.h[self.L - 1]) * (self.xi[self.L - 1] - y)
        for lay in range(self.L - 2, 0, -1):
            self.delta[lay] = self.activation_function.derivative(self.h[lay]) * np.dot(self.w[lay + 1].T, self.delta[lay + 1])

    def __update(self):
        # Update weights and biases based on calculated errors
        for lay in range(1, self.L):
            self.d_w[lay] = -self.lr * np.outer(self.delta[lay], self.xi[lay-1]) + self.momentum * self.d_w_prev[lay]
            self.d_w_prev[lay] = self.d_w[lay].copy()
            self.d_theta[lay] = self.lr * self.delta[lay] + self.momentum * self.d_theta_prev[lay]
            self.d_theta_prev[lay] = self.d_theta[lay].copy()
            self.w[lay] = self.w[lay] + self.d_w[lay]
            self.theta[lay] = self.theta[lay] + self.d_theta[lay]

    def __error(self, X, y, indices):
        # Calculate error for a given set of patterns and targets
        total_error = 0
        for i in indices:
            self.xi[0] = X[i]
            self.__forward()
            total_error += np.sum((self.xi[-1] - y[i]) ** 2)
        return total_error / 2

    def fit(self, X, y):
        # Initialize weights and bias randomly
        for lay in range(1, self.L):
            self.w[lay] = np.random.rand(self.n[lay], self.n[lay - 1])
            self.theta[lay] = np.random.rand(self.n[lay])

        # Split the data into training and validation sets
        total_size = len(X)
        train_size = int((1 - self.valid) * total_size)
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        for epoch in range(self.epochs):
            # Use (1 - self.valid) of the data for training
            for i in train_indices:
                pattern, target = X[i], y[i]
                self.xi[0] = pattern
                self.__forward()
                self.__backward(target)
                self.__update()

            # Feed-forward all training patterns and calculate their prediction quadratic error
            error_train = self.__error(X, y, train_indices)
            self.error_training.append(error_train)
            print("Training - epoch: ", epoch, "Error: ", error_train)

            # Feed-forward all validation patterns and calculate their prediction quadratic error
            error_val = self.__error(X, y, val_indices)
            self.error_validation.append(error_val)
            print("Validation - epoch: ", epoch, "Error: ", error_val)

    def predict(self, X):
        # Predict the output for a given set of patterns
        predictions = []
        for pattern in X:
            self.xi[0] = pattern
            self.__forward()
            predictions.append(self.xi[-1])
        return np.array(predictions)

    def loss_epochs(self):
        # Return training and validation errors for each epoch
        return self.error_training, self.error_validation
