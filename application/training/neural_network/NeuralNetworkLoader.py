import numpy as np


class Layer:

    def __init__(self, name, hocon):
        self.name = name
        self.count = hocon.get_int("count")
        self.skipList = hocon.get_list("skip-list")


class NNLoader:

    def __init__(self, conf):
        self.trainData = []

        # Load train data
        if conf.get_bool("input.is-numpy-array"):
            self.trainData = np.load("../../" + conf.get_string("input.file"), allow_pickle=True)

        # load input feature details
        self.noOfFeaures = conf.get_int("input.no-of-features")
        print("number of features:", self.noOfFeaures)

        print("Training data size:", len(self.trainData))
        if (len(self.trainData[0]) != self.noOfFeaures):
            print("features count did not match")

        self.learningRt = conf.get_float("learning-rate")

        # load hidden layer details
        self.layers = []
        layerConfig = conf.get_config("hidden-layers")
        for layerName in layerConfig:
            self.layers.append(Layer(layerName, layerConfig.get_config(layerName)))
        print(self.layers)

        # load output details
        self.outputType = conf.get_string("output.type")
        self.outputCount = conf.get_int("output.count")
        self.trainOutput = np.load("../../" + conf.get_string("output.file"), allow_pickle=True)

        # load Test details
        self.testData = np.load("../../" + conf.get_string("test.input.file"), allow_pickle=True)
        self.testOutput = np.load("../../" + conf.get_string("test.output.file"), allow_pickle=True)
        self.epsilon = conf.get_float("test.epsilonPerc")

        print("train data shape:", self.trainData.shape, "output size", self.trainOutput.size)
        print("test data shape:", self.testData.shape, "output size", self.testOutput.size)
