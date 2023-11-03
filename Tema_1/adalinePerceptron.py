import numpy as np


class Perceptron:

    def __init__(self, train_data: np.ndarray, train_labels: np.array(np.int8), expected_output: np.uint):
        self.train_data = train_data
        self.expected_output = expected_output
        self.train_labels = self.normalize_labels(train_labels)
        self.weights = np.random.rand(np.shape(train_data)[1], 1)
        self.bias = np.random.rand(1, 1)

    def normalize_labels(self, train_labels: np.ndarray) -> np.ndarray:

        normalized_labels = np.ndarray(train_labels.shape)

        for i in range(train_labels.shape[0]):
            if train_labels[i] != self.expected_output:
                normalized_labels[i] = -1
            else:
                normalized_labels[i] = 1

        return normalized_labels

    def train(self):
        batch_size = 1000
        epoch = 1
        epsilon = 0.0001
        while True:
            learning_rate = get_learning_rate_for_this_epoch(epoch)
            if learning_rate < epsilon:
                break
            self.shuffle_data()
            for i in range(0, self.train_data.shape[0], batch_size):
                (to_update_weights, to_update_biases) = self.__train_batch(self.train_data[i:i + batch_size],
                                                                           self.train_labels[i:i + batch_size],
                                                                           learning_rate)
                self.weights = self.weights + to_update_weights
                self.bias = self.bias + to_update_biases

            epoch += 1

    def shuffle_data(self):
        combined_data = np.concatenate((self.train_data, self.train_labels.reshape(-1, 1)), axis=1)
        np.random.shuffle(combined_data)
        self.train_data = combined_data[:, :-1]
        self.train_labels = combined_data[:, -1].astype(np.int8)

    def __train_batch(self, train_data: np.ndarray, train_labels: np.array(np.int8), learning_rate) -> (
            np.ndarray, np.ndarray):

        mini_batch_size = 100
        shuffle_data(train_data, train_labels)

        to_update_weights = np.zeros(self.weights.shape)
        to_update_biases = 0

        for i in range(train_data.shape[0]):
            (weights_modification, biases_modification) = self.__train_with_one(
                train_data[i].reshape(1, np.shape(train_data)[1]), train_labels[i], learning_rate)

            to_update_weights += weights_modification
            to_update_biases += biases_modification

        return to_update_weights, to_update_biases

    def __train_with_one(self, train_data: np.ndarray, train_label: np.int8, learning_rate) -> (np.ndarray, np.float32):

        neuron_value_raw = np.dot(train_data, self.weights) * train_label + self.bias
        neuron_value = stable_sigmoid(neuron_value_raw)

        if train_label == self.expected_output:
            error = 1 - neuron_value
        else:
            error = 0 - neuron_value

        weights_modification = (train_data * learning_rate * error).reshape(self.weights.shape)
        bias_modification = learning_rate * error

        return weights_modification, bias_modification

    def get_perception(self, data: np.ndarray) -> float:
        return stable_sigmoid(np.dot(data, self.weights) + self.bias)

    def get_accuracy(self):
        correct_predictions = 0
        for data in zip(self.train_data, self.train_labels):
            perception = self.get_perception(data[0])
            if perception > 0.5:
                if data[1] == self.expected_output:
                    correct_predictions += 1
            else:
                if data[1] != self.expected_output:
                    correct_predictions += 1

        return correct_predictions / self.train_data.shape[0]


def stable_sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def get_learning_rate_for_this_epoch(epoch):
    initial_lr = 0.5
    k = 0.2  # Adjust this value for the desired rate of decay
    return initial_lr * 1 / (1 + np.exp(k * epoch))


def shuffle_data(data: np.ndarray, labels: np.array(np.int8)):
    combined_data = np.concatenate((data, labels.reshape(-1, 1)), axis=1)
    np.random.shuffle(combined_data)
    data = combined_data[:, :-1]
    labels = combined_data[:, -1].astype(np.int8)
    return data, labels
