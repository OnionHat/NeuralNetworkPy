from keras.datasets import mnist
from matplotlib import pyplot
from numpy.core.numeric import outer

from neuralnetwork import NeuralNetwork


def main():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # for i in range(9): 
    #     pyplot.subplot(330 + 1 + i)
    #     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    # pyplot.show()


    nn = NeuralNetwork(28*28, 10, 10)

    nn.randomize()
    nn.set_activation("sigmoid")
    nn.set_learningrate(0.01)

    # Fist number is 5
    print(train_X[0].shape)
    for i in range(len(train_X)):
        guess = nn.train(train_X[i].flatten(), train_y[i])
        print(f"Guess: {guess}, Answer: {train_y[i]}")


if __name__ == "__main__":
    main()
