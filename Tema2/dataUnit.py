import numpy as np


class DataUnit:
    def __init__(self, pixel_values: np.array(np.float32), label: np.short):
        self.pixelValues = pixel_values
        self.label = label
