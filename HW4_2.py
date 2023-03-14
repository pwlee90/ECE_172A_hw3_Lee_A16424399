import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

def computeNormGrayHistogram(img):
    #Read image in grayscale mode
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #find sum of each values
    sum = np.zeros(256)
    for i in gray:
        for j in i:
            sum[int(j)] = sum[int(j)] + 1
    output = getNbins(32, sum)
    return output

def getNbins(nbins, sum):
    bitSize = int(len(sum)/nbins)
    # print("sum")
    # print(sum)
    output = np.zeros(nbins)
    for index, i in enumerate(output):
        for j in range(0, bitSize):
            output[index] += sum[int(index * bitSize + j)]
    #normalize
    # print("output")
    # print(output)
    # total = float(0)
    # for i in output:
    #     total += float(i)
    total = output.sum()
    for index, i in enumerate(output):
        output[index] = float(i/total)
    # print("normalized output")
    # print(output)
    return output
def mean_filter(img, padSize):
    output = np.pad(img, (padSize, padSize), 'reflect')
    for x in range(len(img)):
        # print("imgR[x]")
        # print(imgR[x])
        for y in img[x]:
            if(x > padSize and y > padSize):
                sum = 0
                for i in range(padSize):
                    for j in range(padSize):
                        sum = sum + img[x+i-int(padSize-1)/2][y+j-int(padSize-1)/2]
                output[x][y] = sum/(padSize*padSize)
    return output

#his of mural_1
# img = cv2.imread('mural_noise1.jpg')
# cv2.imshow("origin", img)
# x = np.linspace(1, 32, num = 32)
# grayH = computeNormGrayHistogram(img)
# plt.bar(x, grayH)
# plt.show()
# plt.close()
#his of mural_2
# img = cv2.imread('mural_noise2.jpg')
# cv2.imshow("origin", img)
# x = np.linspace(1, 32, num = 32)
# grayH = computeNormGrayHistogram(img)
# plt.bar(x, grayH)
# plt.show()
# plt.close()

#4. mean filters
img = cv2.imread('mural_noise1.jpg', 0)
output = mean_filter(img,5)
plt.plot(output)
plt.show()
plt.close()

#8 template.jpg