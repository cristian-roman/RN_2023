import numpy as np

from dataUnit import DataUnit


class AdalineLoadingPerceptron:
    def __init__(self, path):
        self.bias = None
        self.weights = None
        self.path = path
        self.__load_model()
        self.learningRate = 0.0000001

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
        return np.dot(data, self.weights) + self.bias

    def GetAccuracy(self, trainData: np.array(DataUnit)):
        correct_predictions = 0
        for dataUnit in trainData:
            prediction = self.GetPerception(dataUnit.pixelValues)
            if prediction[0] is True and dataUnit.label == self.expectedOutput:
                correct_predictions += 1
            elif prediction[0] is False and dataUnit.label != self.expectedOutput:
                correct_predictions += 1
            # print(f'Prediction: {prediction[1]}')

        accuracy = correct_predictions / trainData.shape[0]

        # print(f'Perceptron {self.expectedOutput}  accuracy: {accuracy}\n')

        return accuracy
