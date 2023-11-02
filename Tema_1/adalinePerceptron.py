import numpy as np


class Perceptron:
    learning_rate = 0.1
    total_outputs = 10

    def __init__(self, train_data: np.ndarray, train_labels: np.ndarray, expected_output: np.uint):
        self.train_data = train_data
        self.expected_output = expected_output
        self.train_labels = self.normalize_labels(train_labels)
        self.weights = np.random.rand(np.shape(train_data)[1] * Perceptron.total_outputs + Perceptron.total_outputs)

    def normalize_labels(self, train_labels: np.ndarray) -> np.ndarray:

        normalized_labels = np.ndarray(train_labels.shape)

        for i in range(train_labels.shape[0]):
            if train_labels[i] != self.expected_output:
                normalized_labels[i] = -1
            else:
                normalized_labels[i] = 1

        return normalized_labels
