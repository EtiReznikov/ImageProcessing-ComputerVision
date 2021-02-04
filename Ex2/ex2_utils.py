from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List
import math


def myID() -> np.int:
    """
    Return my ID
    :return: int
    """
    return 315674028

def conv1D(inSignal: np.ndarray,kernel1: np.ndarray)->np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """

    kernel_size=np.size(kernel1)
    img_size=np.size(inSignal)
    size=img_size+kernel_size-1
    ans= np.zeros(size)

    np.size(kernel1)

    for x in range(0,size):
        val=0
        for i in range (0,np.size(kernel1)):
            inp_ind=x-i
            if ((inp_ind<img_size and inp_ind>=0) and (i>=0 and i<kernel_size)):
                val+=inSignal[inp_ind]*kernel1[i]
        ans[x]=val
    return ans
    pass



def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """

    image_row, image_col = np.shape(inImage)
    kernel_row, kernel_col = np.shape(kernel2)

    output = np.zeros(np.shape(inImage))

    pad_height=(kernel_row - 1) //2
    pad_width=(kernel_col - 1)//2
    padded_image= np.pad(inImage, (pad_height, pad_width), 'symmetric')
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(padded_image[row:row+kernel_row, col:col+kernel_col]*kernel2)
    return output
    pass


def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """

    k_x= np.array([[1,0,-1]])
    k_y=np.array([[1],[0],[-1]])
    d_x= cv2.filter2D(inImage,-1, k_x)
    d_y= cv2.filter2D(inImage,-1, k_y)
    #
    magnitude = np.square(d_x**2+d_y**2)

    directions= np.arctan2( d_y, d_x)

    return directions, magnitude, d_x, d_y
    pass

def blurImage1(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    2:param kernelSize: Kernel size
    :return: The Blurred image
    """
    sigma = 0.3 * ((kernel_size- 1) * 0.5 - 1) + 0.8
    center = (int)(kernel_size/ 2)
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            diff = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    kernel=kernel/ kernel.sum()
    img= conv2D(in_image,kernel)
    return img

def blurImage2(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    sigma = 0.3 * ((kernel_size- 1) * 0.5 - 1) + 0.8
    d1_gaussian= cv2.getGaussianKernel(kernel_size,sigma)
    kernel = np.multiply(d1_gaussian.T, d1_gaussian)
    img = cv2.filter2D(in_image, -1, kernel)
    return img

def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """

    cv_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    cv_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    cv_result=np.hypot( cv_x, cv_y)

    k_x= np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)/8
    k_y=np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float)/8


    d_x= cv2.filter2D(img, -1, cv2.flip(k_x,0))
    d_y=  cv2.filter2D(img, -1, cv2.flip(k_y,0))

    magnitude= np.sqrt(d_x * d_x + d_y * d_y)
    # magnitude= func_result/ np.max (func_result)

    magnitude = np.float_(magnitude)
    # magnitude *= 255.0 / np.max(magnitude)
    magnitude = magnitude / np.max(magnitude)
    magnitude = magnitude * 255
    thresh= 255*thresh
    func_result= magnitude
    func_result[func_result<thresh]=0
    func_result[func_result>=thresh]=255

    return cv_result,  func_result
    pass

def edgeDetectionZeroCrossingLOG(img:np.ndarray)->(np.ndarray):
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param I: Input image
    :return: :return: Edge matrix
    """

    LOG= np.array([[0,0,-1,0, 0],
                   [0,-1,-2,-1, 0],
                   [-1,-2,16,-2,-1],
                   [0,-1,-2,-1, 0],
                   [0,0,-1,0, 0]])


    img = cv2.filter2D(img, -1, np.flip(LOG))

    return ZeroCrossing(img)
    pass


def edgeDetectionZeroCrossingSimple(img:np.ndarray)->(np.ndarray):
    """
    Detecting edges using the "ZeroCrossing" method
    :param I: Input image
    :return: Edge matrix
    """

    laplacian= np.array([[0,1,0],
                         [1,-4,1],
                         [0,1,0]])
    img= cv2.filter2D(img, -1, laplacian)

    return ZeroCrossing(img)
    pass


def ZeroCrossing(img:np.ndarray)->(np.ndarray):
    """
    Find the zero crossing in image
    :param I: Input image
    :return: The image after zero crossing
    """

    ans = np.zeros(np.shape(img))
    rows, cols = np.shape(img)

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            patch = img[y - 1:y + 2, x - 1:x + 2]
            p = img[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                if minP < 0:
                    zeroCross = True
                else:
                    zeroCross = False
            else:
                if maxP > 0:
                    zeroCross = True
                else:
                    zeroCross = False
            if zeroCross:
                ans[y, x] = 1

    return ans
    pass

def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float)-> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    #blur image ussing gausiian
    blur = cv2.GaussianBlur(img, (5, 5), 0)


    #get directons and mangitude acrodding to sobel
    cv_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    cv_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude=np.hypot( cv_x, cv_y)
    directions= np.arctan2( cv_y, cv_x)
    directions= np.degrees(directions)


    #apply non maximum suppression
    final_image= non_maximum_suppression(magnitude, directions)

    #apply non thresh hold and hysteresis
    final_image= threshold_and_hysteresis(final_image, thrs_1, thrs_2)

    #cv2 solution
    canny_cv2 = cv2.Canny(img,thrs_1,thrs_2)
    return canny_cv2, final_image
    pass


def non_maximum_suppression (mag: np.ndarray, directions: np.ndarray)-> (np.ndarray):
    """
    Non-maximum suppression
    Args:
    img: Gradient magnitude of image for each pixel
    D: Gradient directions of image for each pixel
    Returns: suppresed 2d numpy array of the angels
    """
    rows, cols= np.shape(mag)
    output= np.zeros((rows,cols), dtype=np.int32)

    #Quantize the gradient directions:
    for row in range(1, rows- 1):
        for col in range(1, cols - 1):
            direction = directions[row, col]
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
               val= max(mag[row, col - 1], mag[row, col + 1])
            elif (22.5 <= direction < 67.5):
                val= max(mag[row - 1, col - 1], mag[row + 1, col + 1])
            elif (67.5 <= direction <112.5):
                val = max(mag[row - 1, col], mag[row + 1, col])
            else:
                val = max(mag[row + 1, col - 1], mag[row - 1, col + 1])

            if mag[row, col] >= val:
                output[row,col]= mag[row, col]
    output = np.multiply(output, 255.0 / output.max())

    return output
    pass

def threshold_and_hysteresis(image, low, high):
    weak=50
    strong=255
    output = np.zeros(np.shape(image))

    #Indices of strong edges
    strong_row, strong_col = np.where(image >= high)
    #Indices of weak edges
    weak_row, weak_col = np.where((image <= high) & (image >= low))


    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    #list of indices to get the neighbors of the pixel
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape

    #for each strong pixel, id one of his neihbors is weak- make the weak pixel to strong and it to the strong list
    while np.size(strong_row):
        x = strong_row[0]
        y = strong_col[0]
        strong_row = np.delete(strong_row, 0)
        strong_col = np.delete(strong_col, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if (new_x >= 0 and new_x < size[0]) and(new_y >= 0 & new_y < size[1]):
                if output[new_x, new_y] == weak:

                    output[new_x, new_y] = strong
                    np.append(strong_row, new_x)
                    np.append(strong_col, new_y)

    #evrey pixel that is not strong- make it to zero
    output[output!= strong] = 0
    return output
    pass



def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)-> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """

   #gwt eadge and gradient directions
    edge_image= edgeDetectionCanny(img,100,200)[1]
    directions= convDerivative(img)[0]
    directions= np.degrees(directions)

    pi=math.pi
    blank_image= np.zeros(np.shape(edge_image))
    array = defaultdict(int)
    # for each radius in range
    for r in range(min_radius, max_radius+1):
        # for each radius in pixel
        for i in range(0, edge_image.shape[0]):
            for j in range(0, edge_image.shape[1]):
                #if there is eadge
                if edge_image[i][j] == 255:
                    #check circle acrodding to direction of this eadge
                    current_direct=[]
                    current_direct.append(directions[i,j]+90)
                    current_direct.append(directions[i, j]- 90)
                    for teta in current_direct:
                    # Calculate a ,b adjust to the circle formula
                        a = int(i - r * math.cos(teta * pi / 180))
                        b = int(j - r * math.sin(teta * pi / 180))
                        array[(a, b, r)] += 1

    circles=[]
    #if there atlest 3 voutes, its a circle
    for key in array:
        if array[key]>= 3:
            circles.append(key)

    return circles
    pass