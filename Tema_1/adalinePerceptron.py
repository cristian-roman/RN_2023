import numpy as np


class Perceptron:
    learning_rate = 0.1
    total_outputs = 10

    def __init__(self, train_data: np.ndarray, train_labels: np.array(np.int8), expected_output: np.uint):
        self.train_data = train_data
        self.expected_output = expected_output
        self.train_labels = self.normalize_labels(train_labels)
        self.weights = np.random.rand(np.shape(train_data)[1] * Perceptron.total_outputs)
        self.biases = np.random.rand(Perceptron.total_outputs)

    def normalize_labels(self, train_labels: np.ndarray) -> np.ndarray:

        normalized_labels = np.ndarray(train_labels.shape)

        for i in range(train_labels.shape[0]):
            if train_labels[i] != self.expected_output:
                normalized_labels[i] = -1
            else:
                normalized_labels[i] = 1

        return normalized_labels

    def train_mini_batching(self):
        mini_batch_size = 500
        for i in range(0, self.train_data.shape[0], mini_batch_size):
            (to_update_weights, to_update_biases) = self.__train(self.train_data[i:i + mini_batch_size],
                                                                 self.train_labels[i:i + mini_batch_size])
            self.weights = self.weights + to_update_weights
            self.biases = self.biases + to_update_biases

    def __train(self, train_data: np.ndarray, train_labels: np.array(np.int8)) -> (np.ndarray, np.ndarray):
        to_update_weights = np.zeros(self.weights.shape)
        to_update_biases = np.zeros(self.biases.shape)

        for i in range(train_data.shape[0]):
            (weights_modification, biases_modification) = self.__train_with_one(train_data[i], train_labels[i])

            to_update_weights += weights_modification
            to_update_biases += biases_modification

        return to_update_weights, to_update_biases

    def __train_with_one(self, train_data: np.ndarray, train_label: np.int8) -> (np.ndarray, np.ndarray):

        neuron_value = np.dot(train_data, self.weights) * train_label + self.biases

        if neuron_value > 0:  # classified correctly
            error = train_label - neuron_value

            weights_modification = train_data * self.learning_rate * error
            biases_modification = self.learning_rate * error

        return weights_modification, biases_modification
