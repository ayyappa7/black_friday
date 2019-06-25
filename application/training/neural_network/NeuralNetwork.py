import time

from training.neural_network.NeuralNetworkLoader import NNLoader
import numpy as np, sys
from numpy import random
from pyhocon import ConfigFactory


def sigmoid(val):
    return np.array(1 / (1 + np.exp(-1 * val)))


def train():
    startTime = time.clock()
    loader = NNLoader(ConfigFactory.parse_file("model_structure.conf"))
    layer1 = loader.layers[0]
    coefficientMatrix1 = random.rand(layer1.getSizeAsOutput(), loader.noOfFeaures)  # size = 10*8
    layer2 = loader.layers[1]
    coefficientMatrix2 = random.rand(layer2.getSizeAsOutput(), layer1.getSizeAsInput())  # size= 5*10
    coefficientMatrix3 = random.rand(loader.outputCount, layer2.getSizeAsInput())  # size = 1*5
    mse = 1
    oldMse = 2
    while (oldMse - mse > 0.0001):
        oldMse = mse;
        coefficientMatrix1Sum = np.zeros((layer1.getSizeAsOutput(), loader.noOfFeaures))
        coefficientMatrix2Sum = np.zeros((layer2.getSizeAsOutput(), layer1.getSizeAsInput()))
        coefficientMatrix3Sum = np.zeros((loader.outputCount, layer2.getSizeAsInput()))
        mse = 0
        for i in range(len(loader.trainData)):
            inputArr = np.transpose(loader.trainData[i])
            outputExp = sigmoid(loader.trainOutput[i])
            # outputExp = (float(loader.output[i]))

            # forward pass
            hidden1Val = sigmoid(np.dot(coefficientMatrix1, inputArr))

            if layer1.addBias:
                hidden2Val = sigmoid(np.dot(coefficientMatrix2, np.append(hidden1Val, 1)))
            else:
                hidden2Val = sigmoid(np.dot(coefficientMatrix2, hidden1Val))

            if layer2.addBias:
                outputVal = sigmoid(np.dot(coefficientMatrix3, np.append(hidden2Val, 1)))
            else:
                outputVal = sigmoid(np.dot(coefficientMatrix3, hidden2Val))

            delta4 = outputVal - outputExp

            g = np.append(np.multiply(hidden2Val, 1 - hidden2Val),1)
            delta3 = np.multiply(np.dot(np.transpose(coefficientMatrix3), delta4), np.transpose(g))
            g = np.append(np.multiply(hidden1Val, 1 - hidden1Val),1)
            delta2 = np.multiply(np.dot(np.transpose(coefficientMatrix2), delta3[:layer2.getSizeAsOutput()]), np.transpose(g))

            # error
            mse += delta4 ** 2

            coefficientMatrix1Sum += np.dot(delta2[:layer1.getSizeAsOutput()], np.transpose(hidden1Val))
            coefficientMatrix2Sum += np.dot(delta3[:layer2.getSizeAsOutput()], np.transpose(hidden2Val))
            coefficientMatrix3Sum += np.dot(delta4, np.transpose(outputVal))

            if i % 1000 == 0:
                sys.stdout.write("Progress {:2.1%}".format(i / 550068) + '\r')

        size = len(loader.trainData)
        coefficientMatrix1 -= (loader.learningRt * coefficientMatrix1Sum / size)
        coefficientMatrix2 -= (loader.learningRt * coefficientMatrix2Sum / size)
        coefficientMatrix3 -= (loader.learningRt * coefficientMatrix3Sum / size)

        mse = mse / (2 * size)
        print(" cost:", mse, ", % reduced:", (oldMse - mse) * 100 / mse)

    coefficientMatrix1.dump("./coeffs/layer01.npy")
    coefficientMatrix2.dump("./coeffs/layer12.npy")
    coefficientMatrix3.dump("./coeffs/layer23.npy")
    print(" -=DONE=- ")
    print(" Time:  ", time.clock()-startTime, "s")


def test():
    coefficientMatrix1 = np.load("./coeffs/layer01.npy", allow_pickle=True)
    coefficientMatrix2 = np.load("./coeffs/layer12.npy", allow_pickle=True)
    coefficientMatrix3 = np.load("./coeffs/layer23.npy", allow_pickle=True)
    loader = NNLoader(ConfigFactory.parse_file("model_structure.conf"))
    success = 0
    total = 0
    layer1 = loader.layers[0]
    layer2 = loader.layers[1]

    for i in range(len(loader.testData)):
        inputArr = np.transpose(loader.testData[i])
        outputExp = loader.testOutput[i]*4980+9330

        # forward pass
        hidden1Val = sigmoid(np.dot(coefficientMatrix1, inputArr))

        if layer1.addBias:
            hidden2Val = sigmoid(np.dot(coefficientMatrix2, np.append(hidden1Val, 1)))
        else:
            hidden2Val = sigmoid(np.dot(coefficientMatrix2, hidden1Val))

        if layer2.addBias:
            outputVal = np.dot(coefficientMatrix3, np.append(hidden2Val, 1))
        else:
            outputVal = np.dot(coefficientMatrix3, hidden2Val)

        outputVal = outputVal*4980+9330

        total += 1
        err = ((outputVal - outputExp) * 100 / outputExp)
        if abs(err) < loader.epsilon:
            success += 1
        if i % 100 == 0:
            sys.stdout.write("Progress {:2.1%}".format(i / 55007) + '\r')
    print("success % :", (success * 100) / total)
    print("success count :", success)


if __name__ == '__main__':
    # train()
    test()
