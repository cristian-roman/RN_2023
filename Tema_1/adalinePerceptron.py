import concurrent.futures

import numpy as np


class Perceptron:

    def __init__(self, train_data: np.ndarray = None, train_labels: np.array(np.int8) = None,
                 expected_output: np.uint = None, weights=None,
                 bias=None):

        if train_data is not None:
            self.train_data = self.normalize_data(train_data)
            self.expected_output = expected_output
            self.train_labels = self.normalize_labels(train_labels)
            self.weights = np.random.rand(np.shape(train_data)[1], 1)
            self.bias = np.random.rand(1, 1)
        else:
            self.weights = weights
            self.bias = bias
            self.expected_output = expected_output

    @staticmethod
    def normalize_data(train_data: np.ndarray) -> np.ndarray:
        normalized_data = np.ndarray(train_data.shape)

        for i in range(train_data.shape[0]):
            normalized_data[i] = train_data[i] / 255

        return normalized_data

    def normalize_labels(self, train_labels: np.ndarray) -> np.ndarray:

        normalized_labels = np.ndarray(train_labels.shape)

        for i in range(train_labels.shape[0]):
            if train_labels[i] != self.expected_output:
                normalized_labels[i] = -1
            else:
                normalized_labels[i] = 1

        return normalized_labels

    def train(self):
        batch_size = 500
        epoch = 1
        epsilon = 0.01
        while True:
            learning_rate = self.get_learning_rate_for_this_epoch(epoch)
            if learning_rate < epsilon or epoch > 20:
                break
            (self.train_data, self.train_labels) = self.shuffle_data(self.train_data, self.train_labels)
            for i in range(0, self.train_data.shape[0], batch_size):
                (to_update_weights, to_update_bias) = self.__train_batch(self.train_data[i:i + batch_size],
                                                                         self.train_labels[i:i + batch_size],
                                                                         learning_rate)
                self.weights = self.weights + to_update_weights
                self.bias = self.bias + to_update_bias

            epoch += 1

    def __train_batch(self, train_data: np.ndarray, train_labels: np.array(np.int8), learning_rate) -> (
            np.ndarray, np.ndarray):

        mini_batch_size = 25
        (train_data, train_labels) = self.shuffle_data(train_data, train_labels)

        to_update_weights = np.zeros(self.weights.shape)
        to_update_bias = 0

        # Split the data into mini-batches
        mini_batches = [(train_data[i:i + mini_batch_size], train_labels[i:i + mini_batch_size])
                        for i in range(0, train_data.shape[0], mini_batch_size)]

        # Function to process a mini-batch
        def process_mini_batch(mini_batch):
            mini_data, mini_labels = mini_batch
            __mini_weights = np.zeros(self.weights.shape)
            __mini_bias = 0

            for i in range(mini_data.shape[0]):
                (weights_modification, biases_modification) = self.__train_with_one(
                    mini_data[i].reshape(1, np.shape(mini_data)[1]), mini_labels[i], learning_rate)

                __mini_weights += weights_modification
                __mini_bias += biases_modification

            return __mini_weights, __mini_bias

        # Use concurrent.futures to run mini-batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(process_mini_batch, mini_batches))

        # Combine the results
        for mini_weights, mini_bias in results:
            to_update_weights += mini_weights
            to_update_bias += mini_bias

        return to_update_weights, to_update_bias

    def __train_with_one(self, train_data: np.ndarray, train_label: np.int8, learning_rate) -> (np.ndarray, np.float32):

        neuron_value = (np.dot(train_data, self.weights) * train_label + self.bias) * train_label
        print(f'Neuron value: {neuron_value} of Perceptron {self.expected_output} and train label: {train_label}')

        weights_modification = np.zeros(self.weights.shape)
        bias_modification = 0

        if neuron_value > 0:
            error = train_label - neuron_value
            with open(f'errors_{self.expected_output}.txt', 'a') as file:
                file.write(f'Error: {error}  LearningRate = {learning_rate}\n')

            weights_modification = (train_data * learning_rate * error).reshape(self.weights.shape)
            bias_modification = learning_rate * error

        return weights_modification, bias_modification

    def get_perception(self, data: np.ndarray):
        neuron_value = np.dot(data, self.weights) + self.bias
        probability = self.stable_sigmoid(neuron_value)

        print(f'Probability: {probability} of Perceptron {self.expected_output} and neuron value: {neuron_value}')

        return_value = -1
        if probability > 0.75:
            return_value = self.expected_output

        return return_value, probability

    def save_model(self, path):
        np.savez(path, weights=self.weights, bias=self.bias, expected_output=self.expected_output)

    @staticmethod
    def load_model(path):
        model = np.load(path)
        return Perceptron(weights=model['weights'], bias=model['bias'], expected_output=model['expected_output'])

    @staticmethod
    def stable_sigmoid(x):
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    @staticmethod
    def get_learning_rate_for_this_epoch(epoch):
        initial_lr = 0.5
        k = 0.1  # Adjust this value for the desired rate of decay
        return initial_lr * 1 / (1 + np.exp(k * epoch))

    @staticmethod
    def shuffle_data(data: np.ndarray, labels: np.array(np.int8)):
        combined_data = np.concatenate((data, labels.reshape(-1, 1)), axis=1)
        np.random.shuffle(combined_data)
        data = combined_data[:, :-1]
        labels = combined_data[:, -1].astype(np.int8)
        return data, labels
