import numpy as np
import cv2

file_name = input("Enter the images file names : ")
img = cv2.imread(file_name, 0)
rows, cols = img.shape
lbp_img = np.zeros(((rows)//2, (cols)//2), np.uint8)

def neighborpixels(img, x, y):
    collect_neighbor =[]
    collect_neighbor.append(img[x - 1, y - 1])
    collect_neighbor.append(img[x, y - 1])
    collect_neighbor.append(img[x + 1, y - 1])
    collect_neighbor.append(img[x + 1, y])
    collect_neighbor.append(img[x + 1, y + 1])
    collect_neighbor.append(img[x, y + 1])
    collect_neighbor.append(img[x - 1, y + 1])
    collect_neighbor.append(img[x - 1, y])
    return collect_neighbor

def thresholded(center, neighbor_p):
    cal_threshold = []
    for i in neighbor_p:
        if(i >= center):
            cal_threshold.append(1)
        else:
            cal_threshold.append(0)
    return cal_threshold

for x in range(1, rows-1, 3):
    for y in range(1, cols-1, 3):
        center = img[x, y]
        neighbor_p = neighborpixels(img, x, y)

        values = thresholded(center, neighbor_p)
        weights = [1, 2, 4, 6, 16, 32, 64, 128]

        res = 0
        for a in range(0, len(values)):
            res += weights[a] * values[a]
        lbp_img.itemset((x//2, y//2), res)

cv2.imshow('images', img)
cv2.imshow('LBP imgages', lbp_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


