import gzip
import numpy as np
import matplotlib.pyplot as plt


class ImageLoader:
    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path

        self.__get_images()
        self.__get_labels()

    def __get_images(self) -> None:
        with gzip.open(self.images_path, 'rb') as f:
            self.data = np.frombuffer(f.read(), np.uint8, offset=16)
        self.data = self.data.reshape(-1, 784)

    def __get_labels(self) -> None:
        with gzip.open(self.labels_path, 'rb') as f:
            self.labels = np.frombuffer(f.read(), np.uint8, offset=8)
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
