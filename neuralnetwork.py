import time
from typing import Callable

import numpy as np

from matrix import Matrix


class Derivative:
    @staticmethod
    def sigmoid(x):
        return x * (1 - x)
        # Activation.sigmoid(x) * (1 - Activation.sigmoid(x))


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def ReLU(x):
        if x > 0:
            return x
        else:
            return 0


def create_weights(cols, rows) -> np.ndarray:
    return np.zeros((cols, rows))


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        # self.ih_layer = Matrix(hidden_size, input_size)  # (4,2)
        # self.ho_layer = Matrix(output_size, hidden_size)  # (1,4)

        self.ih_weights = create_weights(hidden_size, input_size)
        self.ho_weights = create_weights(output_size, hidden_size)

        self.act_func: Callable  # activation function
        self.der_func: Callable  # derivative function

        self.learningrate: float = 0.0

    def randomize(self):
        self.ih_weights = np.random.uniform(
            -1.0, 1.0, (self._hidden_size, self._input_size)
        )
        self.ho_weights = np.random.uniform(
            -1.0, 1.0, (self._output_size, self._hidden_size)
        )

    def set_activation(self, func: str):
        if func == "sigmoid":
            self.act_func = Activation.sigmoid
            self.der_func = Derivative.sigmoid

    def activation(self, layer: np.ndarray) -> np.ndarray:
        for i, values in enumerate(layer):
            layer[i] = self.act_func(values)
        return layer

    def derivative(self, layer: np.ndarray):
        for i, values in enumerate(layer):
            layer[i] = self.der_func(values)
        return layer

    """
    TODO: Add a batch system to feedforwad

        if the @param input is a matrix do np.dot(np.array(input).T, weights)
        else do np.dot(weights, input), this is because the first element of matrix 
        need to match second element of matrix, say you have 4 inputs and 3 batch
        the shape would be (3,4) and the weights would be (n,4), where n is the size
        the connected layer so you have to transpose input to (4,3)
    """

    def feedforwad(self, inputs: np.ndarray) -> np.ndarray:
        # Stop the program if the input list is not the same size as input_size
        # assert (
        #     len(inputs_list) == self._input_size
        # ), f"Wrong size inputs, layer {self._input_size} != list {len(inputs_list)}"

        # inputs = np.array(inputs_list, ndmin=2).T

        hidden: np.ndarray = np.dot(self.ih_weights, inputs)  # (4,2) (3,2)
        hidden = self.activation(hidden)

        output: np.ndarray = np.dot(self.ho_weights, hidden)
        output = self.activation(output)

        return output

    def set_learningrate(self, learningrate: float):
        self.learningrate = learningrate

    def train(self, inputs, targets):
        # assert (
        #     len(lable) == self._output_size
        # ), f"The lable and output does not have the same size, output {self._output_size} != lable {len(lable)}"

        if type(inputs) != np.ndarray:
            inputs = np.array(inputs, ndmin=2).T
        if type(targets) != np.ndarray:
            targets = np.array(targets, ndmin=2).T

        hidden: np.ndarray = np.dot(self.ih_weights, inputs)
        outputs: np.ndarray = self.feedforwad(inputs)

        # Calculating output layer errors
        targets = np.array(targets, ndmin=2).T
        output_errors = targets - outputs

        # d_W = lr * E * (o - (1 - o))
        # gradiant from outputs to hidden
        ho_gradiant = (output_errors * outputs * (1.0 - outputs)) # derivative sigmoid
        # ho_gradiant = np.dot( (output_errors * outputs * (1.0 - outputs)), np.transpose(hidden))
        ho_gradiant *= self.learningrate

        ho_delta: np.ndarray = ho_gradiant.reshape(-1, 1) * self.ho_weights.T
        self.ho_weights += ho_delta

        # Calculating hidden layer errors
        hidden_errors: np.ndarray = np.dot(self.ho_weights.T, output_errors)

        # gradiant from hidden to inputs
        ih_gradiant = np.dot(
            (hidden_errors * hidden * (1.0 - hidden)), np.transpose(inputs)
        )
        ih_gradiant *= self.learningrate

        # ih_delta: np.ndarray = ih_gradiant.reshape(-1, 1) * self.ih_weights
        self.ih_weights += ih_gradiant

        # Finds the largest output activation and returns it
        highest_output = np.sort(outputs.flatten())
        return highest_output

        


if __name__ == "__main__":
    np.random.seed(0)  # for debug purpurses
    # np.set_printoptions(precision=15)

    nn = NeuralNetwork(2, 3, 2)

    nn.randomize()
    nn.set_activation("sigmoid")
    nn.set_learningrate(0.01)

    inputs = [0, 1]
    targets = [0, 1]

    start_t = time.time()
    for i in range(10000):

        nn.train(inputs, targets)
        end_t = time.time()
        time_taken = (end_t - start_t)*1000
        start_t = end_t
        print(f"Iteration: ", i)
        print(f"Time: ", time_taken)

