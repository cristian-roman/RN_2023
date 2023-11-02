import numpy as np

import imageLoader as imgLoader
import adalinePerceptron as PerceptronModel

train_images_path = 'dataset/train-images-idx3-ubyte.gz'
train_labels_path = 'dataset/train-labels-idx1-ubyte.gz'

test_images_path = 'dataset/t10k-images-idx3-ubyte.gz'
test_labels_path = 'dataset/t10k-labels-idx1-ubyte.gz'

if __name__ == '__main__':
    trainImages = imgLoader.ImageLoader(train_images_path, train_labels_path)

    perceptron = PerceptronModel.Perceptron(trainImages.data, trainImages.labels, np.uint(0))
    perceptron.train_mini_batching()
