import copy

import numpy as np

import adalineLoadingPerceptron
import concurrent.futures
import imageLoader as imgLoader
import adalinePerceptron as p

train_images_path = 'dataset/train-images-idx3-ubyte.gz'
train_labels_path = 'dataset/train-labels-idx1-ubyte.gz'

test_images_path = 'dataset/t10k-images-idx3-ubyte.gz'
test_labels_path = 'dataset/t10k-labels-idx1-ubyte.gz'

if __name__ == '__main__':

    trainingImagesLoader = imgLoader.ImageLoader(train_images_path, train_labels_path)
    trainingData = trainingImagesLoader.getDataUnitList()

    perceptronList = []
    trustFactors = dict()


    # def train_perceptron(i):
    #     perceptron = p.Perceptron(trainingData, np.uint(i))
    #     perceptron.Train()
    #     perceptron.save_model(f'models/perceptron_{i}.json')
    #     print(f'Perceptron {i} trained')
    #
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.map(train_perceptron, [8])

    def load_perceptron(i):
        perceptron = adalineLoadingPerceptron.AdalineLoadingPerceptron(f'models/perceptron_{i}.json.npz', trainingData)
        perceptronList.append(perceptron)


    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(load_perceptron, range(0, 10))

    testImages = imgLoader.ImageLoader(test_images_path, test_labels_path)

    while True:
        correct_predictions = 0
        for perceptron in perceptronList:
            accuracy = perceptron.GetAccuracy()
            trustFactors[int(perceptron.expectedOutput)] = accuracy

        for (data, label) in zip(testImages.data, testImages.labels):
            guess = []
            totalGuess = 0
            for perceptron in perceptronList:
                answer = perceptron.GetPerception(data)
                if answer[0] is True:
                    guess.append((perceptron.expectedOutput, answer[1]))
                    totalGuess += answer[1]

            if len(guess) == 0:
                continue

            maxTrustLevel = -1
            finalGuess = -1
            for (guessLabel, guessValue) in guess:
                guessPercentage = (guessValue / totalGuess)
                trustLevel = 1 * guessPercentage
                if trustLevel > maxTrustLevel:
                    maxTrustLevel = trustLevel
                    finalGuess = guessLabel

            if finalGuess == label:
                correct_predictions += 1

        accuracy = correct_predictions / testImages.data.shape[0]

        if accuracy > 0.75:
            print(f'Final accuracy: {accuracy}')
            break

        else:
            print(f'Accuracy: {accuracy}')


            def train_perceptron(i):
                perceptron = perceptronList[i]
                perceptron.Train()
                perceptron.save_model(f'models/perceptron_{i}.json')
                print(f'Perceptron {i} trained')


            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(train_perceptron, i) for i in range(10)]
                for future in concurrent.futures.as_completed(futures):
                    pass  # Wait for each future to complete
