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

def computeNormRGBHistogram(img):
   B, G, R = cv2.split(img)
   #flatten B, G, R
   B = np.array(B)
   G = np.array(G)
   R = np.array(R)
   B = B.flatten()
   G = G.flatten()
   R = R.flatten()
   output_b = getNbins(32, B)
   output_g = getNbins(32, G)
   output_r = getNbins(32, R)
   output = []
   output = np.append(output, output_r)
   output = np.append(output, output_g)
   output = np.append(output, output_b)
   output = output.flatten()
#    print("output")
#    print(output)
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
    total = float(0)
    for i in output:
        total += float(i)
    for index, i in enumerate(output):
        output[index] = float(i/total)
    # print("normalized output")
    # print(output)
    return output

img = cv2.imread('forest.jpg')
grayH = computeNormGrayHistogram(img)
plt.plot(grayH)
rgbH = computeNormRGBHistogram(img)
plt.plot(rgbH[0:32])
plt.show()