# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as lin
import math

data1 = []; data2 = []; data3 = []

# 평균벡터
mVClass1 = np.zeros((4, 4));    mVClass2 = np.zeros((4, 4));    mVClass3 = np.zeros((4, 4))

# 공분산벡터
covV1 = np.zeros((4, 4));   covV2 = np.zeros((4, 4));   covV3 = np.zeros((4, 4))

confusionM = np.zeros((3, 3))

# 데이터 불러오는 함수
def readTrainFile():
    f = open('Iris_train.dat')
    for line in f.read().split('\n'):
        data = line.split()
        if(data != []):
            if(data[4] == '1'):
                data1.append(data)
            elif(data[4] == '2'):
                data2.append(data)
            else:
                data3.append(data)

# 평균 벡터를 구하는 함수
def meanVector(dataClass):
    mean = np.sum(dataClass, axis=0)[:4]/40
    return mean

# 공분산 벡터를 구하는 함수
def cov(classData, m):
    covV = np.zeros((4, 4)).astype(float)
    for i in range(0, 4):
        for j in range(0, 4):
            data1 = classData[:,i] - m[i]
            data2 = (classData[:, j] - m[j]).T
            covV[i][j] = np.sum(np.dot(data1, data2))/39
    return covV

# class를 결정하는 함수
def dicision(test):
    temp1 = calDicision(test, mVClass1, covV1)
    temp2 = calDicision(test, mVClass2, covV2)
    temp3 = calDicision(test, mVClass3, covV3)

    return compare(temp1, temp2, temp3)

# decision boundary 계산
def calDicision(test, m, c):
    com1 = np.dot(np.dot(test.T, -0.5 * lin.inv(c)), test)
    com2 = np.dot(np.dot(lin.inv(c), m).T, test)
    com3 =  -0.5 * np.dot(np.dot(m, lin.inv(c)), m.T) - 0.5  * math.log(lin.det(c))

    return com1 + com2 + com3

# 3개 숫자 대소비교
def compare(temp1, temp2, temp3):
    return  (3 if (temp2 < temp3) else 2) if (temp1 < temp2)  else ( 3 if (temp1 < temp3) else 1)

# DATA READ
readTrainFile()
train1 = np.array(data1).astype(float)
train2 = np.array(data2).astype(float)
train3 = np.array(data3).astype(float)

# MEAN VCOTOR
mVClass1 = meanVector(train1)
mVClass2 = meanVector(train2)
mVClass3 = meanVector(train3)

# COVARIANCE VECTOR
covV1 = cov(train1, mVClass1)
covV2 = cov(train2, mVClass2)
covV3 = cov(train3, mVClass3)

# TEST DATA READ
test = []
f = open('Iris_test.dat')
for line in f.read().split('\n'):
    data = line.split()
    if (data != []):
        data = line.split()
        test.append(data)

test = np.array(test).astype(float)

# discriminant function
for testData in test:
    if(testData[4] == 1):
        result = dicision(testData[:4])
        confusionM[result - 1, int(testData[4]) - 1 ] += 1
    elif(testData[4] == 2):
        result = dicision(testData[:4])
        confusionM[result - 1, int(testData[4]) - 1] += 1
    else:
        result = dicision(testData[:4])
        confusionM[result - 1, int(testData[4]) - 1 ] += 1

print("mean class 1");  print(mVClass1)
print("covariance class 1");    print(covV1);   print()
print("mean class 2");  print(mVClass2)
print("covariance class 2");    print(covV2);   print()
print("mean class 3");  print(mVClass3)
print("covariance class 3");    print(covV3);   print()

#print("Train 1")
#print(train1)
#print("Train 2")
#print(train2)
#print("Train 3")
#print(train3)

print("Confusion Matrix")
print(confusionM)

