from pyhocon import ConfigFactory
import numpy as np, sys
from numpy import random


class Layer:

    def __init__(self, name, hocon):
        self.name = name
        self.count = hocon.get_int("count")
        self.skipList = hocon.get_list("skip-list")


class NNLoader:

    def __init__(self, conf):
        self.data = []

        if conf.get_bool("input.is-numpy-array"):
            self.data = np.load("../../" + conf.get_string("input.file"), allow_pickle=True)
        self.noOfFeaures = conf.get_int("input.no-of-features")
        print("number of features:", self.noOfFeaures)
        print("Training data size:", len(self.data))
        if (len(self.data[0]) != self.noOfFeaures):
            print("features count did not match")
        self.learningRt = conf.get_float("learning-rate")
        self.layers = []
        layerConfig = conf.get_config("hidden-layers")
        for layerName in layerConfig:
            self.layers.append(Layer(layerName, layerConfig.get_config(layerName)))
        print(self.layers)

        self.outputType = conf.get_string("output.type")
        self.outputCount = conf.get_int("output.count")
        self.output = np.load("../../" + conf.get_string("output.file"), allow_pickle=True)
        print(self.output.shape)


loader = NNLoader(ConfigFactory.parse_file("model_structure.conf"))
layer1 = loader.layers[0]
coefficientMatrix1 = random.rand(layer1.count, loader.noOfFeaures) / 10  # size = 10*8
layer2 = loader.layers[1]
coefficientMatrix2 = random.rand(layer2.count, layer1.count) / 10  # size= 5*10
coefficientMatrix3 = random.rand(loader.outputCount, layer2.count) / 10  # size = 1*5
mse = 1

while (mse > 0.0001):
    coefficientMatrix1Sum = np.zeros((layer1.count, loader.noOfFeaures))
    coefficientMatrix2Sum = np.zeros((layer2.count, layer1.count))
    coefficientMatrix3Sum = np.zeros((loader.outputCount, layer2.count))
    mse = 0
    for i in range(len(loader.data)):
        inputArr = loader.data[i]
        outputExp = (float(loader.output[i]) - 9330) / 4980

        # forward pass
        hidden1 = np.dot(inputArr, np.transpose(coefficientMatrix1))
        hidden2 = np.dot(hidden1, np.transpose(coefficientMatrix2))
        output = np.dot(hidden2, np.transpose(coefficientMatrix3))

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
    size = len(loader.data)
    coefficientMatrix1 -= (loader.learningRt * coefficientMatrix1Sum / size)
    coefficientMatrix2 -= (loader.learningRt * coefficientMatrix2Sum / size)
    coefficientMatrix3 -= (loader.learningRt * coefficientMatrix3Sum / size)
    print(" cost:" , np.sqrt(mse) / (2 * size))

if __name__ == '__main__':
    loader = NNLoader(ConfigFactory.parse_file("model_structure.conf"))
