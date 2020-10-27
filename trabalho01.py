from skimage.exposure import rescale_intensity
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

#argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-s", "--show", required=False,
	help="True to show the images during execution")
args = vars(ap.parse_args())

#reading image in grayscale mode
image = cv2.imread(args["image"],cv2.IMREAD_GRAYSCALE)
if(args["show"]=="True"):
    cv2.imshow("original image", image)

#kernel definitions
h1 = np.array([[0,0,-1,0,0],
              [0,-1,-2,-1,0],
              [-1,-2,16,-2,-1],
              [0,-1,-2,-1,0], 
              [0,0,-1,0,0]])

h2 = np.array([[1,4,6,4,1],
              [4,16,24,16,4],
              [6,24,36,24,6],
              [4,16,24,16,4], 
              [1,4,6,4,1]])
h2 = h2/256

h3 = np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,1]])

h4 = np.array([[-1,-2,-1],
              [0,0,0],
              [1,2,1]])

h5 = np.array([[-1,-1,-1],
              [-1,9,-1],
              [-1,-1,-1]])

h6 = np.array([[1,1,1],
              [1,1,1],
              [1,1,1]])
h6 = h6/9

h7 = np.array([[-1,-1,2],
              [-1,2,-1],
              [2,-1,-1]])

h8 = np.array([[2,-1,-1],
              [-1,2,-1],
              [-1,-1,2]])

h9 = np.identity(9)

h9 = h9/9

h10 = np.array([[-1,-1,-1,-1,-1],
              [-1,2,2,2,-1],
              [-1,2,9,2,-1],
              [-1,2,2,2,-1], 
              [-1,-1,-1,-1,-1]])
h10 = h10/9

h11 = np.array([[-1,-1,0],
              [-1,0,1],
              [0,1,1]])

#kernel flip
h1 =  np.fliplr(np.flipud(h1))
h2 =  np.fliplr(np.flipud(h2))
h3 =  np.fliplr(np.flipud(h3))
h4 =  np.fliplr(np.flipud(h4))
h5 =  np.fliplr(np.flipud(h5))
h6 =  np.fliplr(np.flipud(h6))
h7 =  np.fliplr(np.flipud(h7))
h8 =  np.fliplr(np.flipud(h8))
h9 =  np.fliplr(np.flipud(h9))
h10 = np.fliplr(np.flipud(h10))
h11 = np.fliplr(np.flipud(h11))

#normalization
def normalizationUint8(image):
    image = rescale_intensity(image, in_range=(0, 255))
    return (image * 255 ).astype(np.uint8)

#filtering with every kernel
image1 = cv2.filter2D(image.astype(np.float64),-1,h1.astype(np.float64))

#normalize the data
image1 = normalizationUint8(image1)
if(args["show"]=="True"):
    cv2.imshow("h1 convole", image1)

#save image
cv2.imwrite('h1convolve.png',image1)

image2 = cv2.filter2D(image.astype(np.float64),-1,h2.astype(np.float64))
#normalize the data

image2 = normalizationUint8(image2)
if(args["show"]=="True"):
    cv2.imshow("h2 convole", image2)
#save image
cv2.imwrite('h2convolve.png',image2)


image3 = cv2.filter2D(image.astype(np.float64),-1,h3.astype(np.float64))

#normalize the data
image3 = normalizationUint8(image3)
if(args["show"]=="True"):
    cv2.imshow("h3 convole", image3)
#save image
cv2.imwrite('h3convolve.png',image3)


image4 = cv2.filter2D(image.astype(np.float64),-1,h4.astype(np.float64))

#normalize the data
image4 = normalizationUint8(image4)
if(args["show"]=="True"):
    cv2.imshow("h4 convole", image4)
#save image
cv2.imwrite('h4convolve.png',image4)

image5 = cv2.filter2D(image.astype(np.float64),-1,h5.astype(np.float64))

#normalize the data
image5 = normalizationUint8(image5)
if(args["show"]=="True"):
    cv2.imshow("h5 convole", image5)
#save image
cv2.imwrite('h5convolve.png',image5)

image6 = cv2.filter2D(image.astype(np.float64),-1,h6.astype(np.float64))

#normalize the data
image6 = normalizationUint8(image6)
if(args["show"]=="True"):
    cv2.imshow("h6 convole", image6)
#save image
cv2.imwrite('h6convolve.png',image6)


image7 = cv2.filter2D(image.astype(np.float64),-1,h7.astype(np.float64))

#normalize the data
image7 = normalizationUint8(image7)
if(args["show"]=="True"):
    cv2.imshow("h7 convole", image7)
#save image
cv2.imwrite('h7convolve.png',image7)


image8 = cv2.filter2D(image.astype(np.float64),-1,h8.astype(np.float64))

#normalize the data
image8 = normalizationUint8(image8)
if(args["show"]=="True"):
    cv2.imshow("h8 convole", image8)
#save image
cv2.imwrite('h8convolve.png',image8)


image9 = cv2.filter2D(image.astype(np.float64),-1,h9.astype(np.float64))

#normalize the data
image9 = normalizationUint8(image9)
if(args["show"]=="True"):
    cv2.imshow("h9 convole", image9)
#save image
cv2.imwrite('h9convolve.png',image9)


image10 = cv2.filter2D(image.astype(np.float64),-1,h10.astype(np.float64))

#normalize the data
image10 = normalizationUint8(image10)
if(args["show"]=="True"):
    cv2.imshow("h10 convole", image10)
#save image
cv2.imwrite('h10convolve.png',image10)


image11 = cv2.filter2D(image.astype(np.float64),-1,h11.astype(np.float64))

#normalize the data
image11 = normalizationUint8(image11)
if(args["show"]=="True"):
    cv2.imshow("h11 convole", image11)
#save image
cv2.imwrite('h11convolve.png',image11)

sobelx = cv2.filter2D(image.astype(np.float64),-1,h4.astype(np.float64))
sobely = cv2.filter2D(image.astype(np.float64),-1,h3.astype(np.float64))
combinedSobel = np.sqrt(np.square(sobelx)+np.square(sobely))
#normalize the data
combinedSobel = normalizationUint8(combinedSobel)

if(args["show"]=="True"):
    cv2.imshow("combinedSobel convole", combinedSobel)
#save image
cv2.imwrite('combinedSobel.png',combinedSobel)

if(args["show"]=="True"):
    cv2.waitKey(0)
    cv2.destroyAllWindows()