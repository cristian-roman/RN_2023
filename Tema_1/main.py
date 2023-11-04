import numpy as np

import imageLoader as imgLoader
import adalinePerceptron as PerceptronModel

train_images_path = 'dataset/train-images-idx3-ubyte.gz'
train_labels_path = 'dataset/train-labels-idx1-ubyte.gz'

test_images_path = 'dataset/t10k-images-idx3-ubyte.gz'
test_labels_path = 'dataset/t10k-labels-idx1-ubyte.gz'

if __name__ == '__main__':

    trainingImagesLoader = imgLoader.ImageLoader(train_images_path, train_labels_path)
    trainingData = trainingImagesLoader.getDataUnitList()

    perceptronList = []

    for i in range(0, 10):
        perceptron = PerceptronModel.Perceptron(trainingData, np.uint(i))
        perceptron.Train()
        perceptron.save_model(f'models/perceptron_{i}.json')
        print(f'Perceptron {i} trained')
        print()
        perceptronList.append(perceptron)

    testImages = imgLoader.ImageLoader(test_images_path, test_labels_path)
    print(f'Test images: {testImages.data.shape[0]}')
    correct_predictions = 0
    for (data, label) in zip(testImages.data, testImages.labels):
        prediction = -1
        prediction_disorder = 100000

        for perceptron in perceptronList:
            perception = perceptron.GetPerception(data)
            if perception[0] and perception[1] < prediction_disorder:
                prediction = perceptron.expectedOutput
                prediction_disorder = perception[1]

        if prediction == label:
            correct_predictions += 1

    print(f'Accuracy: {correct_predictions / testImages.data.shape[0]}')
