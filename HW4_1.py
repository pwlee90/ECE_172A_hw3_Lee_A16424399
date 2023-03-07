import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

def computeNormGrayHistogram(img):
    #Read image in grayscale mode
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #find sum of each values
    sum = np.zeros(256)
    for i in gray:
        sum[int(i)] += 1
    output = getNbins(32, sum)
    return output

def computeNormRGBHistogram(img):
   B, G, R = cv2.split(img)
   output_b = getNbins(32, B)
   output_g = getNbins(32, G)
   output_r = getNbins(32, R)
   output = []
   output.append(output_r, output_g, output_b)
   return output

def getNbins(nbins, sum):
    bitSize = len(sum)/nbins
    output = np.zeros(nbins)
    for i in output:
        for j in bitSize:
            i += sum[int(i * bitSize + j)]
    #normalize
    total = 0
    for i in output:
        total += i
    for i in output:
        i = i/total
    return output

img = cv2.imread('forest.jpg')
grayH = computeNormGrayHistogram(img)
plt.plot(grayH)
rgbH = computeNormGrayHistogram(img)
plt.plot(rgbH[0:32])