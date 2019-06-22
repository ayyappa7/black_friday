from training.neural_network.NeuralNetworkLoader import NNLoader
import numpy as np, sys
from numpy import random
from pyhocon import ConfigFactory



def train():
    loader = NNLoader(ConfigFactory.parse_file("model_structure.conf"))
    layer1 = loader.layers[0]
    coefficientMatrix1 = random.rand(layer1.count, loader.noOfFeaures) / 10  # size = 10*8
    layer2 = loader.layers[1]
    coefficientMatrix2 = random.rand(layer2.count, layer1.count) / 10  # size= 5*10
    coefficientMatrix3 = random.rand(loader.outputCount, layer2.count) / 10  # size = 1*5
    mse = 1
    oldMse = 2
    while (oldMse - mse > 0.00000001):
        oldMse = mse;
        coefficientMatrix1Sum = np.zeros((layer1.count, loader.noOfFeaures))
        coefficientMatrix2Sum = np.zeros((layer2.count, layer1.count))
        coefficientMatrix3Sum = np.zeros((loader.outputCount, layer2.count))
        mse = 0
        for i in range(len(loader.trainData)):
            inputArr = loader.trainData[i]
            outputExp = loader.trainOutput[i]
            # outputExp = (float(loader.output[i]))

            # forward pass
            hidden1 = np.dot(inputArr, np.transpose(coefficientMatrix1))
            hidden2 = np.dot(hidden1, np.transpose(coefficientMatrix2))
            output = np.dot(hidden2, np.transpose(coefficientMatrix3))
            # output[0] = output[0]*4980+9330
            # back propagation
            delta4 = output - outputExp
            g = np.multiply(hidden2, 1 - hidden2)
            delta3 = np.multiply(np.dot(np.transpose(coefficientMatrix3), delta4), np.transpose(g))
            g = np.multiply(hidden1, 1 - hidden1)
            delta2 = np.multiply(np.dot(np.transpose(coefficientMatrix2), delta3), np.transpose(g))

            # error
            mse += delta4 ** 2

            coefficientMatrix1Sum += np.dot(delta2, np.transpose(hidden1))
            coefficientMatrix2Sum += np.dot(delta3, np.transpose(hidden2))
            coefficientMatrix3Sum += np.dot(delta4, np.transpose(output))

            if i % 1000 == 0:
                sys.stdout.write("Progress {:2.1%}".format(i / 550068) + '\r')

        size = len(loader.trainData)
        coefficientMatrix1 -= (loader.learningRt * coefficientMatrix1Sum / size)
        coefficientMatrix2 -= (loader.learningRt * coefficientMatrix2Sum / size)
        coefficientMatrix3 -= (loader.learningRt * coefficientMatrix3Sum / size)

        mse = np.sqrt(mse) / (2 * size)
        print(" cost:", mse, ", % reduced:", (oldMse - mse) * 100 / mse)

    coefficientMatrix1.dump("./coeffs/layer01.npy")
    coefficientMatrix2.dump("./coeffs/layer12.npy")
    coefficientMatrix3.dump("./coeffs/layer23.npy")
    print(" -=DONE=- ")


def test():
    coefficientMatrix1 = np.load("./coeffs/layer01.npy", allow_pickle=True)
    coefficientMatrix2 = np.load("./coeffs/layer12.npy", allow_pickle=True)
    coefficientMatrix3 = np.load("./coeffs/layer23.npy", allow_pickle=True)
    loader = NNLoader(ConfigFactory.parse_file("model_structure.conf"))
    success = 0
    total = 0
    for i in range(len(loader.testData)):
        outputExp = loader.testOutput[i]
        inputArr = loader.testData[i]
        hidden1 = np.dot(inputArr, np.transpose(coefficientMatrix1))
        hidden2 = np.dot(hidden1, np.transpose(coefficientMatrix2))
        output = np.dot(hidden2, np.transpose(coefficientMatrix3))
        total += 1
        err = ((output - outputExp) * 100 / outputExp)
        if abs(err) < loader.epsilon:
            success += 1
        if i % 100 == 0:
            sys.stdout.write("Progress {:2.1%}".format(i / 55007) + '\r')
    print("success % :", (success * 100) / total)
    print("success count :", success )


if __name__ == '__main__':
    train()
    test()
