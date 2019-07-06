import csv, time, sys
import numpy as np

prodIdN = 'Product_ID'
genderN = 'Gender'
ageN = 'Age'
occpN = 'Occupation'
cityCatN = 'City_Category'
stayCCN = 'Stay_In_Current_City_Years'
marrN = 'Marital_Status'
prodCat1N = 'Product_Category_1'
purchN = 'Purchase'

productIdMap = {}
AgeMap = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
occupationSize = 20
city = {'A': 1, 'B': 2, 'C': 3}
stay = {'0': 1, '1': 2, '2': 3, '3': 4, '4+': 5}


def getScaledValues(row):
    vector = np.zeros((1, 8))
    vector[0][0] = productIdMap[row[prodIdN]] / 3623

    if row[genderN] == 'M':
        vector[0][1] = 1

    vector[0][2] = AgeMap[row[ageN]] / 6

    vector[0][3] = (float(row[occpN]) - 8.08) / 6.52  # normalised

    vector[0][4] = city[row[cityCatN]] / 3

    vector[0][5] = stay[row[stayCCN]] / 5

    vector[0][6] = float(row[marrN])

    vector[0][7] = (float(row[prodCat1N]) - 5.3) / 3.75

    purhaseVal = (float(row[purchN]) - 9330) / 4980
    return vector, purhaseVal


def preprocessTrainData():
    scaledVector = np.zeros((0, 8))
    output = np.zeros(0)
    with open("../../application/data/train.csv", "rt") as f:
        data = csv.DictReader(f)
        lastID = 1
        i = 0
        for row in data:
            dictData = dict(row)

            # populating productID map
            if dictData[prodIdN] not in productIdMap.keys():
                newID = dictData[prodIdN]
                productIdMap[newID] = lastID
                lastID += 1

            vect, y = getScaledValues(dictData)
            scaledVector = np.append(scaledVector, vect, axis=0)
            output = np.append(output, y)
            i += 1
            if i % 1000 == 0:
                sys.stdout.write('\r' + "Progress {:2.1%}".format(i / 550068))

    return scaledVector, output


def preprocessAllData():
    scaledTrainVector = np.zeros((0, 8))
    scaledTestVector = np.zeros((0, 8))
    trainOutput = np.zeros(0)
    testOutput = np.zeros(0)
    with open("../../application/data/train.csv", "rt") as f:
        data = csv.DictReader(f)
        lastID = 1
        i = 0
        for row in data:
            dictData = dict(row)

            # populating productID map
            if dictData[prodIdN] not in productIdMap.keys():
                newID = dictData[prodIdN]
                productIdMap[newID] = lastID
                lastID += 1

            vect, y = getScaledValues(dictData)
            if i % 10 == 0:
                scaledTestVector = np.append(scaledTestVector, vect, axis=0)
                testOutput = np.append(testOutput, y)
            else:
                scaledTrainVector = np.append(scaledTrainVector, vect, axis=0)
                trainOutput = np.append(trainOutput, y)
            i += 1
            if i % 1000 == 0:
                sys.stdout.write('\r' + "Progress {:2.1%}".format(i / 550068))

    return scaledTrainVector, trainOutput, scaledTestVector, testOutput


if __name__ == '__main__':
    scaledTrainVector, trainOutput, scaledTestVector, testOutput = preprocessAllData()
    scaledTrainVector.dump("../../application/data/scaled_own_train_22_jun.npy")
    scaledTestVector.dump("../../application/data/scaled_own_test_22_jun.npy")
    trainOutput.dump("../../application/data/output_own_train_22_jun.npy")
    testOutput.dump("../../application/data/output_own_test_22_jun.npy")
    scaledVector2 = np.load("../../application/data/scaled_own_train_22_jun.npy", allow_pickle=True)
    scaledVector3 = np.load("../../application/data/scaled_own_test_22_jun.npy", allow_pickle=True)
    print(scaledVector2.shape)
    print(scaledVector3.shape)
