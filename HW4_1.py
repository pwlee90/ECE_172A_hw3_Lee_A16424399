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
    output = np.empty([np.shape(im)[0],np.shape(im)[1]])
    #flatten B, G, R
    im = np.array(im)
    padSize = int((winSize-1)/2)
    # print("padSize")
    # print(padSize)
    
    #Pad the image im on all 4 sides by mirroring intensity values so that the contextual region for edge pixels remains valid.
    #### not padding!
    
    imgR = cv2.copyMakeBorder(im,padSize,padSize+1,padSize,padSize+1,cv2.BORDER_REPLICATE)
    # print("np.shape(imgR)")
    # print(np.shape(imgR))
    # print(imgR)
    # print("imgR")
    #imgG = cv2.copyMakeBorder(G, padSize, padSize, padSize, padSize, cv2.BORDER_CONSTANT)
    # print("imgR")
    # print(imgR)
    #for each pixel (x, y) in im, do
    for x in range(np.shape(im)[0]):
        # print("imgR[x]")
        # print(imgR[x])
        for y in range(np.shape(im)[1]):
            if(x > padSize and y > padSize):
                # print("y")
                # print(y)
                rank = 0
                #contextual region = winSize x winSize window centered around (x, y)
                region = []
                region = np.array(region)
                xrange = range(x-int(winSize/2) , x+int(winSize/2))
                yrange = range(y-int(winSize/2) , y+int(winSize/2))
                val = imgR[xrange,:][:,yrange]
                # print("val")
                # print(val)
                # for (i, j) in contextual region do
                for i in range(np.shape(val)[0]):
                    for j in range(np.shape(val)[1]):
                            #if im(x, y) > im(i, j) then
                            # print("val[0][int(winSize/2)]")
                            # print(val[0][int(winSize/2)])
                            if(val[int(winSize/2)][int(winSize/2)] > val[i][j]):
                                rank = rank + 1
                    #output(x, y) = rank x 255/(winSize x winSize)
                    output[x][y] = rank * 255 / (winSize * winSize)
    # print("output")
    # print(output)
    return output
#1. 2. 3. 
#plot the histogram
img = cv2.imread('forest.jpg')
cv2.imshow("origin", img)
x = np.linspace(1, 32, num = 32)
grayH = computeNormGrayHistogram(img)
plt.bar(x, grayH)
plt.show()
plt.close()
# rgb = computeNormRGBHistogram(img)
# color = np.zeros(0)
# for i in range(0, 32):
#     color = np.append(color, 'red')
# for i in range(0, 32):
#     color = np.append(color, 'green') 
# for i in range(0, 32):
#     color = np.append(color, 'blue') 
# color_list = color.tolist()  
# # # print("color list")
# # # print(color_list)
# x = np.linspace(1, 96, num = 96)
# plt.bar(x, rgb, color = color_list)
# plt.show()
# plt.close()

#4. 
# # #plot the flipped image's histogram
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

#5. 
#double the values of R
# (B, G, R) = cv2.split(img)
# # # print("R")
# # # print(R)
# for index, i in enumerate(R):
#     R[index] = i*2
# # # print("new R")
# # # print(R)
# new_img = cv2.merge([B, G, R])
# cv2.imshow("2Rforest", new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# grayH = computeNormGrayHistogram(new_img)
# x = np.linspace(1, 32, num = 32)
# plt.bar(x, grayH)
# plt.show()
# plt.close()
# rgb = computeNormRGBHistogram(new_img)
# color = np.zeros(0)
# for i in range(0, 32):
#     color = np.append(color, 'red')
# for i in range(0, 32):
#     color = np.append(color, 'green') 
# for i in range(0, 32):
#     color = np.append(color, 'blue') 
# color_list = color.tolist()  
# x = np.linspace(1, 96, num = 96)
# plt.bar(x, rgb, color = color_list)
# plt.show()
# plt.close()

# 6. AHE
# he_img = cv2.imread('beach.png',0)
# # print("np.size(he_img)")
# # print(np.shape(he_img))
# ahe = AHE(he_img, 2)
# # print("ahe")
# # print(ahe)
# cv2.imshow("ahe - test", ahe)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#7. HE
# he = cv2.equalizeHist(he_img) 
# cv2.imshow("he", he)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
