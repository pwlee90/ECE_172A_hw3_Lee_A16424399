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
def mean_filter(img, winSize):
    img = np.array(img)
    padSize = int((winSize-1)/2)
    print("padSize")
    print(padSize)
    output = np.empty([np.shape(img)[0],np.shape(img)[1]])
    #Pad the image im on all 4 sides by mirroring intensity values so that the contextual region for edge pixels remains valid.
    #### not padding!
    
    imgR = cv2.copyMakeBorder(img,padSize,padSize+1,padSize,padSize+1,cv2.BORDER_REPLICATE)
    for x in range(np.shape(img)[0]):
        # print("imgR[x]")
        # print(imgR[x])
        for y in range(np.shape(img)[1]):
            sum = np.sum(imgR[x:x+padSize,:][:,y:y+padSize])
            output[x][y] = sum/(padSize*padSize)/255
    return output
def median_filter(img, winSize):
    img = np.array(img)
    padSize = int((winSize-1)/2)
    print("padSize")
    print(padSize)
    output = np.empty([np.shape(img)[0],np.shape(img)[1]])
    #Pad the image im on all 4 sides by mirroring intensity values so that the contextual region for edge pixels remains valid.
    #### not padding!
    
    imgR = cv2.copyMakeBorder(img,padSize,padSize+1,padSize,padSize+1,cv2.BORDER_REPLICATE)
    for x in range(np.shape(img)[0]):
        # print("imgR[x]")
        # print(imgR[x])
        for y in range(np.shape(img)[1]):
            median = np.median(imgR[x:x+padSize,:][:,y:y+padSize])
            output[x][y] = median/255
    return output
def minimal_value(img, original_img):
    #for every point in original image as the top left point
    max  = 0
    max_x = 0
    max_y = 0
    for x in range(len(original_img) - len(img)):
        for y in range(len(original_img[0]) - len(img[0])):
            sum = 0
            #for every point in cropped image
            for i in range(len(img)):
                for j in img[i]:
                    sum += abs(original_img[x][y] - img[i][j])
            if sum > max:
                max = sum
                max_x = x
                max_y = y
    return max_x, max_y
    

#his of mural_1
# img1 = cv2.imread('mural_noise1.jpg')
# cv2.imshow("origin", img1)
# x = np.linspace(1, 32, num = 32)
# grayH = computeNormGrayHistogram(img1)
# plt.bar(x, grayH)
# plt.show()
# plt.close()
# # his of mural_2
# img2 = cv2.imread('mural_noise2.jpg')
# cv2.imshow("origin", img2)
# x = np.linspace(1, 32, num = 32)
# grayH = computeNormGrayHistogram(img2)
# plt.bar(x, grayH)
# plt.show()
# plt.close()

#4. mean filters
#mural 1
#mean
# img1 = cv2.imread('mural_noise1.jpg', 0)
# # print(np.shape(img1))
# # print("np.shape(img)")
# output = mean_filter(img1,81)
# # print("np.shape(output)")
# # print(np.shape(output))
# cv2.imshow("mean 1", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#median
# img1 = cv2.imread('mural_noise1.jpg', 0)
# # print(np.shape(img1))
# # print("np.shape(img)")
# output = median_filter(img1,81)
# # print("np.shape(output)")
# # print(np.shape(output))
# cv2.imshow("median 1", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#mural 2
#mean
# img2 = cv2.imread('mural_noise2.jpg', 0)
# # print(np.shape(img1))
# # print("np.shape(img)")
# output = mean_filter(img2,5)
# # print("np.shape(output)")
# # print(np.shape(output))
# cv2.imshow("mean 2", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#median
# img2 = cv2.imread('mural_noise2.jpg', 0)
# # print(np.shape(img1))
# # print("np.shape(img)")
# output = median_filter(img2,81)
# # print("np.shape(output)")
# # print(np.shape(output))
# cv2.imshow("median 2", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#8 find crop image in original iamge
# Read the template
# template = cv2.imread('template', 0)
# # Store width and height of template in w and h
# w, h = template.shape[::-1]
# original_img = cv2.imread('template_old.jpeg', 0)
# E_i, E_j = minimal_value(template, original_img)
# # 9 perfect match, the f^2 term should equal the t^2 term
# # 10 cross-correlation
# res = cv2.matchTemplate(original_img, template, 'cv2.TM_CCORR')
# # Store the coordinates of matched area in a numpy array
# threshold = 0.8
# loc = np.where(res >= threshold)
# # Draw a rectangle (100,100) center the matched region.
# center_x = int(range(loc)/2)
# center_y = int(range(loc[0])/2)
# cv2.rectangle(original_img, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (0, 255, 255), 2)
# # Show the final image with the matched area.
# cv2.imshow('Detected', original_img)
# # 11 normalized cross-correlation
# res = cv2.matchTemplate(original_img, template, 'cv2.TM_CCORR_NORMED')

#ndimage.convolve
#dot product the smallest
#max is the closest pattern
#normalize mean = 0, varaince = 1
#sqrt(sum(square ))= variance

#library
# patches(matplotlib.patches)
# rescale (skiimage.transform)
# scripy(ndimage)