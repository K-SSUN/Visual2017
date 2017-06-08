# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as lin
import math
import matplotlib.pyplot as plt

data1 = []; data2 = []; data3 = []

# 평균벡터
mVClass1 = np.zeros((2, 2));    mVClass2 = np.zeros((2, 2));    mVClass3 = np.zeros((2, 2))

# 공분산벡터
covV1 = np.zeros((2, 2));   covV2 = np.zeros((2, 2));   covV3 = np.zeros((2, 2))

trainX = []
trainY = []
confusionM = np.zeros((3, 3))

# 데이터 불러오는 함수
def readFile(fileName):
    f = open(fileName)
    global data1; global data2; global data3;
    for line in f.read().split('\n'):
        data = line.split()
        temp = []
        if(data != []):
            if(data[4] == '1'):
                temp = data[:2];    temp.append(str(data[4]))
                data1.append(temp)
            elif(data[4] == '2'):
                temp = data[:2];    temp.append(str(data[4]))
                data2.append(temp)
            else:
                temp = data[:2];    temp.append(str(data[4]))
                data3.append(temp)

# 평균 벡터를 구하는 함수
def meanVector(dataClass):
    mean = np.sum(dataClass, axis=0)[:2]/40
    return mean

# 공분산 벡터를 구하는 함수
def cov(classData, m):
    covV = np.zeros((2, 2)).astype(float)
    for i in range(0, 2):
        for j in range(0, 2):
            data1 = classData[:,i] - m[i]
            data2 = (classData[:, j] - m[j]).T
            covV[i][j] = np.sum(np.dot(data1, data2))/39
    return covV

# Mahalanobis distance = 2
def mahalanobis(x, y, m, c):
    covT = lin.inv(c)
    return (x-m[0])*(covT[0][0]*(x-m[0])+covT[1][0]*(y-m[1])) + \
           (y-m[1])*(covT[0][1]*(x-m[0])+covT[1][1]*(y-m[1])) - 2

# decision boundary를 결정하는 함수
def decisionBoundary(x, y, m1, c1, m2, c2):
    cT1 = lin.inv(c1)
    com1 = -0.5*((cT1[0][0]*x+cT1[1][0]*y)*x + (cT1[1][0]*x+cT1[1][1]*y)*y)
    com2 = x*(cT1[0][0]*m1[0]+cT1[0][1]*m1[1]) + y*(cT1[1][0]*m1[0]+cT1[1][1]*m1[1])
    com3 = -0.5*(m1[0]*(cT1[0][0]*m1[0]+cT1[1][0]*m1[1])+m1[1]*(cT1[0][1]*m1[0]+cT1[1][1]*m1[1])) \
           -0.5*math.log(lin.det(c1))
    g1 = com1 + com2 + com3

    cT2 = lin.inv(c2)
    com1 = -0.5 * ((cT2[0][0] * x + cT2[1][0] * y) * x + (cT2[1][0] * x + cT2[1][1] * y) * y)
    com2 = x * (cT2[0][0] * m2[0] + cT2[0][1] * m2[1]) + y * (cT2[1][0] * m2[0] + cT2[1][1] * m2[1])
    com3 = -0.5 * (m2[0] * (cT2[0][0] * m2[0] + cT2[1][0] * m2[1]) + m2[1]
                   * (cT2[0][1] * m2[0] + cT2[1][1] * m2[1])) - 0.5 * math.log(lin.det(c2))
    g2 = com1 + com2 + com3

    return g1 - g2

# decision boundary함수를 이용하여 class 결정
def decision(x, y):

    return (3 if decisionBoundary(x, y, mVClass3, covV3, mVClass1, covV1) > 0 else 1) \
        if (decisionBoundary(x, y, mVClass1, covV1, mVClass2, covV2) > 0)  \
        else (2 if decisionBoundary(x, y, mVClass2, covV2, mVClass3, covV3) > 0 else 3)

# DATA READ
readFile('Iris_train.dat')
train1 = np.array(data1).astype(float)
train2 = np.array(data2).astype(float)
train3 = np.array(data3).astype(float)

#print("Train 1")
#print(train1)
#print("Train 2")
#print(train2)
#print("Train 3")
#print(train3)

# MEAN VCOTOR
mVClass1 = meanVector(train1)
mVClass2 = meanVector(train2)
mVClass3 = meanVector(train3)

# COVARIANCE VECTOR
covV1 = cov(train1, mVClass1)
covV2 = cov(train2, mVClass2)
covV3 = cov(train3, mVClass3)


# plot
x = np.linspace(4, 8.3, 100)
y = np.linspace(1.8, 4.7, 100)
X,Y = np.meshgrid(x,y)

plt.figure()

# mahalanobis = 2
w1 = plt.contour(X, Y, mahalanobis(X,Y, mVClass1, covV1), 0, colors='red')
plt.clabel(w1, fontsize=10, inline=1,fmt = 'w1')
w2 = plt.contour(X, Y, mahalanobis(X,Y, mVClass2, covV2), 0, colors='blue')
plt.clabel(w2, fontsize=10, inline=1,fmt = 'w2')
w3 = plt.contour(X, Y, mahalanobis(X,Y, mVClass3, covV3), 0, colors='green')
plt.clabel(w3, fontsize=10, inline=1,fmt = 'w3')

g12 = plt.contour(X, Y, decisionBoundary(X, Y, mVClass1, covV1, mVClass2, covV2), 0, colors='red')
plt.clabel(g12, fontsize=10, inline=1,fmt = 'g12')
g23 = plt.contour(X, Y, decisionBoundary(X, Y, mVClass2, covV2, mVClass3, covV3), 0, colors='green')
plt.clabel(g23, fontsize=10, inline=1,fmt = 'g23')
g31 = plt.contour(X, Y, decisionBoundary(X, Y, mVClass3, covV3, mVClass1, covV1), 0, colors='blue')
plt.clabel(g31, fontsize=10, inline=1,fmt = 'g31')

# train set
trainX1 = train1[:, 0]; trainY1 = train1[:, 1]
trainX2 = train2[:, 0]; trainY2 = train2[:, 1]
trainX3 = train3[:, 0]; trainY3 = train3[:, 1]
plt.plot(trainX1, trainY1, 'r+')
plt.plot(trainX2, trainY2, 'b+')
plt.plot(trainX3, trainY3, 'g+')

# mean
plt.plot(mVClass1[0], mVClass1[1], 'ro')
plt.plot(mVClass2[0], mVClass2[1], 'bo')
plt.plot(mVClass3[0], mVClass3[1], 'go')

print("mean class 1");  print(mVClass1)
print("covariance class 1");    print(covV1);   print()
print("mean class 2");  print(mVClass2)
print("covariance class 2");    print(covV2);   print()
print("mean class 3");  print(mVClass3)
print("covariance class 3");    print(covV3);   print()

# TEST DATA READ
data1 = []; data2 = []; data3 = []
readFile('Iris_test.dat')
test1 = np.array(data1).astype(float)
test2 = np.array(data2).astype(float)
test3 = np.array(data3).astype(float)

# test set
testX1 = test1[:, 0]; testY1 = test1[:, 1]
testX2 = test2[:, 0]; testY2 = test2[:, 1]
testX3 = test3[:, 0]; testY3 = test3[:, 1]

for i in test1, test2, test3:
    for j in i :
        if(j[2] == 1):
            result = decision(j[0], j[1])
            confusionM[result - 1, int(j[2]) - 1] += 1
            if(result == int(j[2])):
                plt.plot(j[0], j[1], 'c^')
            else:
                plt.plot(j[0], j[1], 'cD')
        elif(j[2] == 2):
            result = decision(j[0], j[1])
            confusionM[result - 1, int(j[2]) - 1] += 1
            if (result == int(j[2])):
                plt.plot(j[0], j[1], 'm^')
            else:
                plt.plot(j[0], j[1], 'mD')
        else:
            result = decision(j[0], j[1])
            confusionM[result - 1, int(j[2]) - 1] += 1
            if (result == int(j[2])):
                plt.plot(j[0], j[1], 'y^')
            else:
                plt.plot(j[0], j[1], 'yD')

print("Confusion Matrix")
print(confusionM)
plt.show()


