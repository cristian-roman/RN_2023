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
        perceptron.save_model(f'models/perceptron_{i}.json')
        print(f'Perceptron {i} trained')
        print()
        perceptronList.append(perceptron)
        #perceptronList.append(PerceptronModel.Perceptron.load_model(f'models/perceptron_{i}.json.npz'))

    testImages = imgLoader.ImageLoader(test_images_path, test_labels_path)
    print(f'Test images: {testImages.data.shape[0]}')
    correct_predictions = 0
    for (data, label) in zip(testImages.data, testImages.labels):
        prediction = -1
        prediction_probability = -1

        for perceptron in perceptronList:
            perception = perceptron.get_perception(data)

            if perception[0] != -1:
                if prediction_probability < perception[1]:
                    prediction = perception[0]
                    prediction_probability = perception[1]

        if prediction == label:
            correct_predictions += 1

    print(f'Accuracy: {correct_predictions / testImages.data.shape[0]}')
