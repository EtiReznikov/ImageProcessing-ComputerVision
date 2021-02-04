"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 315674028


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns in converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    try:
        #Reads the file and converts it to the requested representation
        image = cv2.imread(filename)
        if representation==1:
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif representation==2:
            image_np= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # invaild input
            print("Representation does not exist")
            exit(1)
        image_np= (image_np-0)/255 #Normalize the numpy array
        return  image_np
    except cv2.error as e:
        # error handling- the file does not exist
        print("file was not found")
        exit(1)
    pass


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    #Converts the image to normalize numpy array
    image_np=imReadAndConvert(filename, representation)
    # Reads the numpy array and converts it to the requested representation
    if representation == 1:
        plt.imshow(image_np, cmap=cm.gray, vmin=0, vmax=1)
    if representation==2:
        plt.imshow(image_np,  vmin=0, vmax=1)
    plt.show()

    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    #Reshape and transpose the image to the appropriate dimensions, multiplies it, and changes the matrix back to its original dimensions.
    new_img = imgRGB.reshape(imgRGB.shape[0]*imgRGB.shape[1], imgRGB.shape[2])
    new_img = new_img.transpose()
    mat = np.array([[0.299, 0.587, 0.114], [0.585, -0.275, -0.321], [0.212, -0.523, 0.311]]) #RGB to YIQ mat
    YIQ= np.dot (mat, new_img)
    ans=YIQ.transpose()
    ans= ans.reshape( imgRGB.shape[0], imgRGB.shape[1],3)
    return ans
    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # Reshape and transpose the image to the appropriate dimensions, multiplies it, and changes the matrix back to its original dimensions.
    new_img = imgYIQ.reshape((imgYIQ.shape[0] * imgYIQ.shape[1]), imgYIQ.shape[2])
    new_img = new_img.transpose()
    mat = np.array([[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]]) #YIQ to RGB mat
    RGB= np.dot (mat, new_img)
    ans = RGB.transpose()
    ans = ans.reshape(imgYIQ.shape[0], imgYIQ.shape[1], 3)
    return ans
    pass

def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """
    #get the representation of the image
    flag= isGrayScale(imgOrig)
    #Produces the histogram according to the representation of the image
    if flag:
        imgOrig=(imgOrig*255)
        imgOrig=imgOrig.astype(int)
        hist, bin_edges = np.histogram(imgOrig, bins=256)

    else:
        yiq_img, Y, hist, bin_edges= getYIQdata(imgOrig)
        # yiq_img = yiq_img.astype(int)
    #Produces cumsum, create LUT and replace any intensity according to LUT
    cdf = np.cumsum(hist, dtype=float)
    LUT = np.zeros(256)
    f_x= (1/cdf[255])*255
    if (flag):
        new_img=imgOrig
        original=imgOrig
    else:
        new_img=Y
        original=Y
    for px in range(0, 256):
        LUT[px]= round(cdf[px]*f_x)
        ind= np.where(original==px)
        new_img[ind]=LUT[px]
    #Creates the returned image and the new histogram according to the given representation.
    if(flag):
        img=new_img
        new_hist, new_bin_edges = np.histogram(img, 256)
    else:
        yiq_img[:, :, 0] = new_img
        img = transformYIQ2RGB(yiq_img)
        img=img/255
        new_hist, new_bin_edges = np.histogram(img[:, :, 0], bins=256)
    return img, hist, new_hist
    pass


def isGrayScale(img : np.ndarray) -> (bool):
    """
        Identify if grayscale or color img loaded
        :param filename: The path to the image
        :return true if the imagr is gray scale, other- false
    """
    #If the image has 3 dimensions, it is RGB
    if len(img.shape)==3:
        return False
    return True
    pass

def getYIQdata(imOrig: np.ndarray):
    """
        Returns YIQ image information data- YIQimg, Y chanel, histogram of y chanle and the bins of the histogram
        :param imOrig: Normlize numpy array of RGB image
        :return (yiq_img, yiq_img[:,:,0], hist, bin_edges)
    """
    yiq_img = transformRGB2YIQ(imOrig)
    yiq_img = yiq_img * 255
    yiq_img = yiq_img.astype(int)
    Y = yiq_img[:, :, 0]
    hist, bin_edges = np.histogram(Y, bins=256)
    return yiq_img, yiq_img[:,:,0], hist, bin_edges


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # invaild input
    if (nQuant<1 or nIter<0):
        print("invaild input")
        exit(1)
    #get the representation of the image
    flag = isGrayScale(imOrig)
    # Produces the histogram and the image, according to the representation of the image
    if (flag):
        imgOrig = (imOrig * 255)
        hist, bin_edges = np.histogram(imgOrig, bins=256)
        img=imgOrig
    else:
        yiq_img, Y, hist, bin_edges = getYIQdata(imOrig)
        imgOrig = (imOrig * 255)
        imgOrig = imgOrig.astype(int)
        img=Y
    hist_p = hist / np.sum(hist)
    hist_p = hist_p / 100

    #Initial division to Z intervals
    Z = [0]
    z_size = 255 / nQuant
    z_size = round(z_size)
    R = z_size
    for x in range(0, nQuant-1):
        Z.append(int(R))
        R += z_size
    Z.append(255)

    # org = img.ravel()
    means = [None] * nQuant
    images= []
    images.append(imgOrig)
    MSE=[]

    for j in range(nIter):
        if (flag):
            image = imOrig
        else:
            image = Y
        image = image.astype(float)
        org = img.ravel()

        for i in range(nQuant):
            # find q's
            index_0 = Z[i]
            index_1 = Z[i + 1] + 1
            q = np.average(range(index_0, index_1), weights=hist_p[index_0:index_1])
            means[i] = q

        #Update the image according to q's
            index_0 = Z[i]
            index_1 = Z[i + 1] + 1
            image = image.ravel()
            for px in range(0, len(image)):
                if org[px] >= index_0 and org[px] < index_1:
                    image[px] = means[i]

        if (flag):
            image = np.reshape(image, np.shape(imgOrig))
            images.append(image)
            #Calculating MSE
            MSE.append(np.square(imgOrig - image).mean())


        else:
            image = np.reshape(image, np.shape(Y))
            #Calculating MSE
            MSE.append(np.square(Y - image).mean())
            #Back from YIQ to RGB
            new_img = yiq_img
            new_img[:, :, 0] = image
            new_img = new_img / 255
            new_img = transformYIQ2RGB(new_img)
            image=new_img
            images.append(new_img)
            images.append(new_img)

        # Testing that MSE is still in decline
        if (j>0):
              if (MSE[j]==MSE[j-1] or MSE[j-1]<MSE[j]):
                  break
        # Update Z's
        for i in range(1, nQuant):
            Z[i] = (means[i - 1] + means[i]) / 2
            Z[i] = int(Z[i])
    return images, MSE
    pass
