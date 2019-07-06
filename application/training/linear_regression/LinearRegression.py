import numpy as np, sys


def train():
    learningRate = 0.5
    coeffs = np.random.rand(9)
    trainData = np.load("../../data/scaled_own_train_22_jun.npy", allow_pickle=True)
    outputData = np.load("../../data/output_own_train_22_jun.npy", allow_pickle=True)
    oldMse = 2
    mse = 1

    while abs(oldMse - mse) > 0.0001:

        oldMse = mse
        se = 0
        error = np.zeros(len(trainData[0]) + 1)
        for i in range(len(trainData)):
            inputArr = trainData[i]
            outputExp = outputData[i]
            output = np.dot(inputArr, np.transpose(coeffs[:8])) + coeffs[8]

            temp = inputArr * (output - outputExp)
            error += np.append(temp, (output - outputExp))
            se += (output - outputExp) ** 2
            if i % 100 == 0:
                sys.stdout.write("Progress {:2.1%}".format(i / 550068) + '\r')

        mse = se / (2 * len(trainData))
        print("cost:", mse, ", % reduced: ", (oldMse - mse) * 100 / oldMse)
        coeffs = coeffs - (learningRate * (error / len(trainData)))

    return coeffs


def test(coeffs):
    testData = np.load("../../data/scaled_own_test_22_jun.npy", allow_pickle=True)
    outputData = np.load("../../data/output_own_test_22_jun.npy", allow_pickle=True)
    success = 0
    for i in range(len(testData)):
        inputArr = testData[i]
        outputExp = outputData[i]
        output = np.dot(inputArr, np.transpose(coeffs[:8])) + coeffs[8]
        if (abs(outputExp - output) * 100 / outputExp) < 1:
            print("exp:", outputExp*4980+9330,"pred:",output*4980+9330)
            success += 1
    print("success rate", success*100/len(testData), "success count: ",success)

# coeffs = train()
# print(coeffs)
# coeffs.dump("coeffs.npy")
coeffs = np.load("coeffs.npy",allow_pickle=True)
test(coeffs)
