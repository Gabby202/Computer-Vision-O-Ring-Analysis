import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt


def threshold(img, thresh):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] > thresh:
                img[i, j] = 0
            else:
                img[i, j] = 255
    return img

def imhist(img):
    hist = np.zeros(256)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            hist[img[i, j]] += 1
    return hist

def findT(hist):
    max = 0
    for i in range(256):
        if hist[i] > max:
            max = hist[i]
            peak = i
    T = peak - 50
    return T

def dilation(img):
    copy = img.copy()
    for i in range(10, img.shape[0] - 10):
        for j in range(10, img.shape[1] - 10):
            if img[i, j] == 0:
                for x in range(i - 2, i + 2):
                    for y in range(j - 2, j + 2):
                        if img[x, y] == 255:
                            copy[i, j] = 255
    img = copy
    return img

def erosion(img):
    copy = img.copy()
    for i in range(10, img.shape[0] - 10):
        for j in range(10, img.shape[1] - 10):
            if img[i, j] == 255:
                for x in range(i - 2, i + 2):
                    for y in range(j - 2, j + 2):
                        if img[x, y] == 0:
                            copy[i, j] = 0
    img = copy
    return img

def label(img):
    # I(i, j)
    copy = img.copy()
    labels = np.zeros((copy.shape[0], copy.shape[1]))
    currlabel = 0
    q = []

    for i in range(copy.shape[0]):
        for j in range(copy.shape[1]):
            if copy[i, j] == 255 and labels[i, j] == 0:
                currlabel += 1
                labels[i, j] = currlabel
                q.append((i, j))  # = [(i, j)]
                while len(q) != 0:
                    pixel = q.pop()
                    labels[pixel[0], pixel[1]] = currlabel
                    if pixel[0] < 0 and img[pixel[0] - 1, pixel[1]] == 0 and labels[pixel[0] - 1, pixel[1]] == 0:
                        q.append((pixel[0] - 1, pixel[1]))
                    if pixel[0] + 1 < img.shape[0] and img[pixel[0] + 1, pixel[1]] == 0 and labels[
                        pixel[0] + 1, pixel[1]] == 0:
                        q.append((pixel[0] + 1, pixel[1]))
                    if pixel[1] < 0 and img[pixel[0], pixel[1] - 1] == 0 and labels[pixel[0], pixel[1] - 1] == 0:
                        q.append((pixel[0], pixel[1] - 1))
                    if pixel[1] + 1 < img.shape[1] and img[pixel[0], pixel[1] + 1] == 0 and labels[
                        pixel[0], pixel[1] + 1] == 0:
                        q.append((pixel[0], pixel[1] + 1))
    img = copy
    return img

def display_labels(labels):
    for i in range(0, labels.shape[0]):
        s = 0
        for j in range(0, labels.shape[1]):
            s += labels[i, j]
        print(s)

for i in range(15):
    img = cv.imread('C:/Users/Gabby/Documents/year4_semester2/Computer Vision/Orings/Oring' + str(i + 1) + '.jpg', 0)
    copy = img.copy()
    hist = imhist(img)
    # Fx = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
    # Fy = np.array([[-1,-1,-1], [0, 0, 0], [1,1,1]])
    # Fxy = np.array([[-2, -1, 0], [-1, 0, 10], [0, 1, 2]])
    T = findT(hist)
    plt.plot(hist)
    # plots line at threshold
    plt.plot((T, T), (0, 500), 'g')
    plt.show()
    before = time.time()
    img = threshold(img, T)
    img = dilation(img)
    img = erosion(img)
    img = label(img)
    after = time.time()
    img = cv.putText(img,"processed: " + str(after-before) + "s",(5, 210), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0))
    cv.imshow("image 1", img)
cv.destroyAllWindows()

