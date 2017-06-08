import cv2 as cv
import numpy as np

img = cv.imread('C:\\Users\\K_SSUN\\PycharmProjects\\visual_lab\\pos\\pos_train\\crop_000001a.png', cv.IMREAD_GRAYSCALE)
rows, cols = img.shape

img = img.astype(np.float64)
Ix = img.copy()
Iy = img.copy()

for i in range(1,rows-2):
    Iy[i,:]=img[i,:]-img[i+2,:] #x축 방향 미분값 저장
for i in range(1,cols-2):
    Ix[:,i]=img[:,i]-img[:,i+2] #y축 방향 미분값 저장

# np.arctan real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2)
# angles in range (0,180)
angle = np.arctan(np.divide(Ix, Iy))
angle = np.rad2deg(angle) + 90.0
magnitude = np.sqrt(np.multiply(Ix, Ix)+np.multiply(Iy, Iy)) # np.multiply = 요소 곱

# Remove redundant pixels in an image.
# angle = angle[~np.isnan(angle)] 이러면 nan을 다 없애버림
for i in range(0, rows):
    for j in range(0, cols):
        if(np.isnan(angle[i, j])): # np.isnan(angle[i, j]) nan이면 return true
            angle[i, j] = 0
        if(np.isnan(magnitude[i, j])):
            magnitude[i, j] = 0

#initialized the feature vector
feature=[];