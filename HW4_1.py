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
   (B, G, R) = cv2.split(img)
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
    # total = float(0)
    # for i in output:
    #     total += float(i)
    total = output.sum()
    for index, i in enumerate(output):
        output[index] = float(i/total)
    # print("normalized output")
    # print(output)
    return output

def AHE(im, winSize):
    output = [[]]
    (B, G, R) = cv2.split(im)
    #flatten B, G, R
    B = np.array(B)
    G = np.array(G)
    R = np.array(R)
    padSize = int((winSize-1)/2)
    print("R length")
    print(len(R))
    #Pad the image im on all 4 sides by mirroring intensity values so that the contextual region for edge pixels remains valid.
    #### not padding!
    imgR = cv2.copyMakeBorder(R, padSize, padSize, padSize, padSize, cv2.BORDER_CONSTANT)
    imgG = cv2.copyMakeBorder(G, padSize, padSize, padSize, padSize, cv2.BORDER_CONSTANT)
    imgB = cv2.copyMakeBorder(B, padSize, padSize, padSize, padSize, cv2.BORDER_CONSTANT)
    # print("imgR")
    # print(imgR)
    #for each pixel (x, y) in im, do
    for x in range(len(imgR)):
        # print("imgR[x]")
        # print(imgR[x])
        for y in imgR[x]:
            # print("y")
            # print(y)
            rank = 0
            #contextual region = winSize x winSize window centered around (x, y)
            region = []
            region = np.array(region)
            for winX in range(winSize):
                list = []
                list = np.array(region)
                for winY in range(winSize):
                    print("imgR[x-winSize/2 + winX]")
                    print(imgR[x-winSize/2 + winX])
                    val = [imgR[x-winSize/2 + winX], imgR[y-winSize/2 + winY]]
                    print("val")
                    print(val)
                    list = np.append(region,val)
                print("list")
                print(list)
                region = np.append(region, list)
                # for (i, j) in contextual region do
                for i in range(len(region)):
                    for j in i:
                        #     if im(x, y) > im(i, j) then
                        if(imgR[x][y] > img[i][j]):
                            rank = rank + 1
                    #output(x, y) = rank x 255/(winSize x winSize)
                    output = np.append(output,[rank * 255 / (winSize * winSize)])
                    output[x][y] = rank * 255 / (winSize * winSize)
    print("output")
    print(output)
    return output
#plot the histogram
img = cv2.imread('forest.jpg')
#cv2.imshow("origin", img)
# x = np.linspace(1, 32, num = 32)
# grayH = computeNormGrayHistogram(img)
# plt.bar(x, grayH)
# plt.show()
# plt.close()
# rgb = computeNormRGBHistogram(img)
# color = np.zeros(0)
# for i in range(0, 32):
#     color = np.append(color, 'red')
# for i in range(0, 32):
#     color = np.append(color, 'green') 
# for i in range(0, 32):
#     color = np.append(color, 'blue') 
# color_list = color.tolist()  
# # print("color list")
# # print(color_list)
# x = np.linspace(1, 96, num = 96)
# plt.bar(x, rgb, color = color_list)
# plt.show()
# plt.close()
# #plot the flipped image's histogram
# img_flip = np.fliplr(img)
# x = np.linspace(1, 32, num = 32)
# grayH = computeNormGrayHistogram(img)
# plt.bar(x, grayH)
# plt.show()
# plt.close()
# rgb = computeNormRGBHistogram(img)
# x = np.linspace(1, 96, num = 96)
# plt.bar(x, rgb, color = color_list)
# plt.show()
# plt.close()
#double the values of R
# (B, G, R) = cv2.split(img)
# # print("R")
# # print(R)
# for index, i in enumerate(R):
#     R[index] = i*2
# # print("new R")
# # print(R)
# new_img = cv2.merge([B, G, R])
# cv2.imshow("2Rforest", new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# AHE
ahe = AHE(img, 2)
print("ahe")
print(ahe)
cv2.imshow("ahe", ahe)
cv2.waitKey(0)
cv2.destroyAllWindows()

