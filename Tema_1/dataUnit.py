import numpy as np


class DataUnit:
    def __init__(self, pixelValues: np.array(np.float32), label: np.short):
        self.pixelValues = pixelValues
        self.label = label
