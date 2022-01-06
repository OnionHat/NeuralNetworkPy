import random


class Layer:
    def __init__(self, size: int):
        self._size: int = size
        self._values: list[float] = [0 for _ in range(size)]
        self._weight: list[list[float]] = []
        self._biases: list[float] = []

    def __str__(self):
        return f"values: {self._values} \nweight: {self._weight}"

    def get_size(self):
        return self._size

    def get_weight_list(self) -> list[list[float]]:
        return self._weight

    def get_weight(self, col: int, row: int) -> float:
        return self._weight[col][row]

    def get_value(self, index) -> float:
        return self._values[index]

    def new_weight_col(self, row: list[float]) -> None:
        self._weight.append(row)

    def set_values(self, values) -> None:
        self._values = values


class NeuralNetwork:
    def __init__(self, input: int, hidden: int, output: int):
        self.input: Layer = Layer(input)
        self.hidden: Layer = Layer(hidden)
        self.output: Layer = Layer(output)

    def init_weight(self, first_layer: Layer, second_layer: Layer) -> None:
        for _ in range(second_layer.get_size()):
            row = []
            for _ in range(first_layer.get_size()):
                value = random.uniform(-1, 1)
                row.append(value)
            second_layer.new_weight_col(row)

    def forward(self, acitvasion_layer: Layer,
                preceptron_layer: Layer, biases: list[float]) -> None:
        # Dot product
        arr = []
        for i in range(preceptron_layer.get_size()):
            values = 0
            for j in range(acitvasion_layer.get_size()):
                values += acitvasion_layer.get_value(index=j) * \
                    preceptron_layer.get_weight(col=i, row=j)
            arr.append(self.ReLu(values + biases[i]))
        preceptron_layer.set_values(values=arr)

    def ReLu(self, x: float) -> float:
        if x < 0:
            return 0
        return x

    # TODO: Learn how backpropogation works and implement it
    # WARN: hell
    # INFO: aa
    # HINT: asdasd
    # TEST: asdsdasds

    def backpropogation(self):
        pass


def main():
    nn = NeuralNetwork(input=4, hidden=3, output=2)
    # input_layer = Layer(4)
    # hidden_layer = Layer(3)
    # output_layer = Layer(2)

    nn.init_weight(nn.input, nn.hidden)
    nn.init_weight(nn.hidden, nn.output)

    nn.input._values[0] = 1.0
    nn.input._values[2] = 3.0
    nn.input._values[1] = 2.0
    nn.input._values[3] = 2.5

    nn.hidden._weight = [[0.2, 0.8, -0.5, 1.0],
                         [0.5, -0.91, 0.26, -0.5],
                         [-0.26, -0.27, 0.17, 0.87]]

    biases = [2.0, 3.0, 0.5]
    # print(dot(input_layer, hidden_layer, biases))

    nn.forward(nn.input, nn.hidden, biases)
    nn.forward(nn.hidden, nn.output, biases)


if __name__ == "__main__":
    main()
