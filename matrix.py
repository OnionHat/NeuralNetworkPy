import numpy as np


class Matrix:
    def __init__(self, cols, rows):
        self._rows = rows
        self._cols = cols
        self._weights: np.ndarray = np.zeros((cols, rows))

    def randomize(self) -> None:
        self._weights = np.random.uniform(-1.0, 1.0, (self._cols, self._rows))

    def get_weights(self) -> np.ndarray:
        return self._weights
