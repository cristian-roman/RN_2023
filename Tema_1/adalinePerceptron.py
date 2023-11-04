import concurrent.futures
import os

import numpy as np

import copy

from dataUnit import DataUnit


class Perceptron:
    # parameters
    batchSize = 200
    miniBatchSize = np.uint8(batchSize / os.cpu_count())
    startingLearningRate = 0.0001
    accuracyEpsilon = 0.0001

    def __init__(self, trainData: np.array(DataUnit), expectedOutput):
        self.expectedOutput = expectedOutput
        # Create a deep copy of trainData
        self.trainData = copy.deepcopy(trainData)

        self.weights = np.random.rand(trainData[0].pixelValues.shape[1], 1)

        self.bias = np.random.rand(1, 1)
        self.lastAccuracies = []

        self.learningRate = self.startingLearningRate
        self.epoch = 0

    def Train(self):
        self.__normalize_data()
        np.random.shuffle(self.trainData)

        while True:
            for batchNumber in range(0, self.trainData.shape[0], self.batchSize):
                (rawWeightsUpdateVector, rawBiasUpdateValue) = self.__train_batch(batchNumber)

                self.weights = self.weights + rawWeightsUpdateVector / self.batchSize
                self.bias = self.bias + rawBiasUpdateValue / self.batchSize

            if self.__isBreakingConditionTriggered():
                break

            self.learningRate = self.learningRate

    def __isBreakingConditionTriggered(self):
        currentAccuracy = self.__get_accuracy()

        if not self.lastAccuracies:
            self.lastAccuracies.append(currentAccuracy)
            return False

        meanAccuracy = np.mean(self.lastAccuracies)
        if currentAccuracy < 0.77 or (currentAccuracy - meanAccuracy) > self.accuracyEpsilon:
            if len(self.lastAccuracies) >= 3:
                self.lastAccuracies[self.epoch % 3] = currentAccuracy
            else:
                self.lastAccuracies.append(currentAccuracy)
            return False
        else:
            return True

    def __get_accuracy(self):
        self.epoch += 1
        correct_predictions = 0
        for dataUnit in self.trainData:
            prediction = self.GetPerception(dataUnit.pixelValues)
            if prediction[0] is True and dataUnit.label == 1:
                correct_predictions += 1
            elif prediction[0] is False and dataUnit.label == -1:
                correct_predictions += 1
            # print(f'Prediction: {prediction[1]}')

        accuracy = correct_predictions / self.trainData.shape[0]

        print(f'Perceptron {self.expectedOutput} epoch {self.epoch} accuracy: {accuracy}\n')

        return accuracy

    def __normalize_data(self):
        for i in range(self.trainData.shape[0]):
            self.trainData[i].pixelValues = self.trainData[i].pixelValues / 255
            self.trainData[i].label = 1 if self.trainData[i].label == self.expectedOutput else -1

    def __train_batch(self, batchNumber: int) -> (
            np.ndarray, np.float32):

        batchData = self.trainData[batchNumber:batchNumber + self.batchSize]

        to_update_weights = np.zeros(self.weights.shape)
        to_update_bias = 0

        # Split the data into mini-batches
        mini_batches = [(batchData[k:k + self.miniBatchSize])
                        for k in range(0, self.batchSize, self.miniBatchSize)]

        # Function to process a mini-batch
        def processMiniBatch(mini_batch):
            __mini_weights = np.zeros(self.weights.shape)
            __mini_bias = 0

            for i in range(mini_batch.shape[0]):
                (weightsModification, biasModification) = self.__train_with_one(mini_batch[i].pixelValues,
                                                                                mini_batch[i].label)
                __mini_weights += weightsModification
                __mini_bias += biasModification

            return __mini_weights, __mini_bias

        # Use concurrent.futures to run mini-batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(processMiniBatch, mini_batches))

        # Combine the results
        for mini_weights, mini_bias in results:
            to_update_weights += mini_weights
            to_update_bias += mini_bias

        return to_update_weights, to_update_bias

    def __train_with_one(self, trainData: np.ndarray, trainLabel: np.short) -> (np.ndarray, np.float32):

        weightsModification = np.zeros(self.weights.shape)
        biasModification = 0

        raw_neuron_value = self.GetRawNeuronValue(trainData)
        if raw_neuron_value * trainLabel < 0:
            if raw_neuron_value < 0:
                error = self.expectedOutput - raw_neuron_value
            else:
                error = - raw_neuron_value
            # print (f'Error: {error}')
            weightsModification = self.learningRate * trainData.reshape(784, 1) * error
            biasModification = self.learningRate * error
        return weightsModification, biasModification

    def GetPerception(self, data: np.ndarray):

        raw_neuron_value = self.GetRawNeuronValue(data)
        if raw_neuron_value >= 0:
            return True, raw_neuron_value
        else:
            return False, raw_neuron_value

    def save_model(self, path):
        np.savez(path, weights=self.weights, bias=self.bias, expected_output=self.expectedOutput)

    def GetRawNeuronValue(self, data: np.ndarray):
        neuron_value = np.dot(data, self.weights) + self.bias
        return neuron_value
