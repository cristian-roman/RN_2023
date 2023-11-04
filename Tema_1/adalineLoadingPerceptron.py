import numpy as np


class AdalineLoadingPerceptron:
    def __init__(self, path):
        self.path = path
        self.__load_model()

    def __load_model(self):
        data = np.load(self.path)
        self.weights = data['weights']
        self.bias = data['bias']
        self.expectedOutput = data['expected_output']

    def GetPerception(self, data: np.ndarray):
        raw_neuron_value = self.GetRawNeuronValue(data)
        if raw_neuron_value >= 0:
            return True, raw_neuron_value
        else:
            return False, raw_neuron_value

    def GetRawNeuronValue(self, data: np.ndarray):
        return np.dot(self.weights.T, data) + self.bias
