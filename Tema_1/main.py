import numpy as np

import imageLoader as imgLoader
import adalinePerceptron as PerceptronModel

train_images_path = 'dataset/train-images-idx3-ubyte.gz'
train_labels_path = 'dataset/train-labels-idx1-ubyte.gz'

test_images_path = 'dataset/t10k-images-idx3-ubyte.gz'
test_labels_path = 'dataset/t10k-labels-idx1-ubyte.gz'

if __name__ == '__main__':
    trainImages = imgLoader.ImageLoader(train_images_path, train_labels_path)

    perceptronList = []
    for i in range(10):
        perceptron = PerceptronModel.Perceptron(trainImages.data, trainImages.labels, np.uint(i))
        perceptron.train()
        print(f'Perceptron {i} trained')
        perceptronList.append(perceptron)

    testImages = imgLoader.ImageLoader(test_images_path, test_labels_path)

    correct_predictions = 0
    for (data, label) in zip(testImages.data, testImages.labels):
        best_prediction = -1
        best_prediction_probability = -1

        for perceptron in perceptronList:
            prediction = perceptron.get_perception(data)

            if prediction > best_prediction_probability:
                best_prediction = perceptron.expected_output
                best_prediction_probability = prediction

        if best_prediction == label:
            correct_predictions += 1

    print(f'Accuracy: {correct_predictions / testImages.data.shape[0]}')
