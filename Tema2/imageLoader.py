import gzip
import numpy as np
import matplotlib.pyplot as plt

from dataUnit import DataUnit


class ImageLoader:
    def __init__(self, imageset_path, labelset_path):
        self.imageSetPath = imageset_path
        self.labelSetPath = labelset_path

        self.__get_images()
        self.__get_labels()

    def getDataUnitList(self):
        dataUnitList = []

        for (pixelValues, label) in zip(self.data, self.labels):
            pixelValues = pixelValues.reshape(1, 784)
            label = label.reshape(1, 1)
            dataUnitList.append(DataUnit(pixelValues, label))

        return np.array(dataUnitList)

    def __get_images(self) -> None:
        with gzip.open(self.imageSetPath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        self.data = data.reshape(-1, 784)

    def __get_labels(self) -> None:
        with gzip.open(self.labelSetPath, 'rb') as f:
            self.labels = np.frombuffer(f.read(), np.int8, offset=8)
        self.labels = self.labels.reshape(-1, 1)

    def show_images(self, lower_bound: int, upper_bound: int) -> None:
        i = 0
        for image in self.data[lower_bound:upper_bound]:
            imageReshaped = image.reshape(28, 28)
            label = self.labels[i][0]

            plt.imshow(imageReshaped, cmap='gray')
            plt.title(f'Label: {label}')
            plt.show()
            i += 1

    def getNumberOfImagesOf(self, i):
        return np.count_nonzero(self.labels == i)
