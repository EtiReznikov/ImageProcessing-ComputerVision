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
    Reads an image, and returns and returns in converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    if representation==1:
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if representation==2:
        image_np= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np= (image_np-0)/255
    return  image_np
    pass


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image_np=imReadAndConvert(filename, representation)
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

    new_img = imgRGB.reshape(imgRGB.shape[0]*imgRGB.shape[1], imgRGB.shape[2])
    new_img = new_img.transpose()
    mat = np.array([[0.299, 0.587, 0.114], [0.585, -0.275, -0.321], [0.212, -0.523, 0.311]])
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
    new_img = imgYIQ.reshape((imgYIQ.shape[0] * imgYIQ.shape[1]), imgYIQ.shape[2])
    new_img = new_img.transpose()
    mat = np.array([[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]])
    RGB= np.dot (mat, new_img)
    ans = RGB.transpose()
    ans = ans.reshape(imgYIQ.shape[0], imgYIQ.shape[1], 3)


    return ans
    pass


def calHist(img: np.ndarray) -> np.ndarray:
    img_flat = img.ravel()
    hist = np.zeros(257)

    for pix in img_flat:
        hist[pix] +=1

    return hist

def calCumSum(arr: np.array) -> np.ndarray:
   cum_sum = np.zeros_like(arr)
   cum_sum[0] = arr[0]
   arr_len = len(arr)
   for idx in range(1, arr_len):
       cum_sum[idx] = arr[idx] + cum_sum[idx - 1]
   return cum_sum


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """
    if isGrayScale(imgOrig):
        imgOrig=(imgOrig*255)
        imgOrig=imgOrig.astype(int)
        hist, bin_edges = np.histogram(imgOrig, bins=256)
        cdf = np.cumsum(hist, dtype=float)
        f = np.zeros(256)
        f_x= (1/cdf[255])*255
        new_img=imgOrig
        for px in range(0, 256):
            f[px]= round(cdf[px]*f_x)
            new_img[new_img==px]= f[px]
        img=new_img
        # print(img)
        new_hist, new_bin_edges=np.histogram(img, 256)
        # return img, hist, new_hist

    else:
        yiq_img = transformRGB2YIQ(imgOrig)
        yiq_img=yiq_img*255
        yiq_img=yiq_img.astype(int)
        Y=yiq_img[:,:,0]
        hist, bin_edges = np.histogram(Y, bins=256)
        cdf = np.cumsum(hist, dtype=float)
        f = np.zeros(256)
        f_x = (1 / cdf[255]) * 255
        new_img = Y

        for px in range(0, 256):
            f[px] = round(cdf[px] * f_x)
            new_img[new_img == px] = f[px]
        # print(new_img)
        yiq_img[:,:,0]=new_img
        # print(yiq_img)

        img= transformYIQ2RGB(yiq_img)
        new_hist ,  new_bin_edges= np.histogram(img[:, :, 0], bins=256)
        img=img/255


    return img,hist, new_hist
    pass



def isGrayScale(img : np.ndarray) -> (bool):
    """
        identify if grayscale or color img loaded
        :param filename: The path to the image
        :return true if the imagr is gray scale, other- false
    """
    if len(img.shape)==3:
        return False

    return True
    pass

def getYhistogram(imOrig: np.ndarray):
    yiq_img = transformRGB2YIQ(imOrig)
    yiq_img = yiq_img * 255
    yiq_img = yiq_img.astype(int)
    Y = yiq_img[:, :, 0]

    hist, bin_edges = np.histogram(Y, bins=256)
    return hist, bin_edges

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if isGrayScale(imOrig):
        imgOrig = (imOrig * 255)
        imgOrig = imgOrig.astype(int)
        hist, bin_edges = np.histogram(imgOrig, bins=256)

        # plt.hist(hist,  bins=255)
        # plt.show()
        # print(hist)
        # cdf = np.cumsum(hist, dtype=float)
        # print (cdf[255])
        # print( np.sum(hist))
        hist_p=hist/np.sum(hist)
        hist_p=hist_p/100

        # plt.hist(hist_p)
        # # plt.show()
        Z=[0]
        z_size=255/nQuant
        z_size=round(z_size)
        R=z_size
        org = imgOrig.ravel()
        means = [None] * nQuant
        images =[None] *(nIter+1)
        images[0]=imgOrig
        MSE =[None] *nIter
        for x in range (0,nQuant):
            Z.append(int(R))
            R+=z_size
        for j in range(nIter):

            for i in range(nQuant):
                index_0= Z[i]
                index_1= Z[i+1]+1
                q=np.average(range(index_0,index_1), weights=hist_p[index_0:index_1])
                means[i]=q
            image = imOrig
            for i in range(nQuant):
                index_0 = Z[i]
                index_1 = Z[i + 1] + 1
                # np.put(image, range(index_0,index_1), means[i])
                image=  image.ravel()

                for px in range(0,len(image)):
                    if org[px]>= index_0 and org[px] <index_1:
                        image[px]=means[i]
                image=np.reshape(image, np.shape(imOrig))
            images[j+1]=image
            # MSE[j] = np.square(np.subtract(imOrig, image)).mean()
            MSE[j]=np.square(imgOrig - image).mean()
            for i in range(1,nQuant):
                Z[i]=(means[i-1]+means[i])/2
                Z[i]=int(Z[i])
    else:

        # plt.imshow(imOrig, vmin=0, vmax=1)
        # plt.show()
        yiq_img = transformRGB2YIQ(imOrig)
        yiq_img = yiq_img * 255
        yiq_img = yiq_img.astype(int)
        Y = yiq_img[:, :, 0]

        imgOrig = (imOrig * 255)
        imgOrig = imgOrig.astype(int)

        hist, bin_edges = np.histogram(Y, bins=256)

        # plt.hist(hist,  bins=255)
        # plt.show()
        # print(hist)
        # cdf = np.cumsum(hist, dtype=float)
        # print (cdf[255])
        # print( np.sum(hist))
        hist_p = hist / np.sum(hist)
        hist_p = hist_p / 100

        # plt.hist(hist_p)
        # # plt.show()
        Z = [0]
        z_size = 255 / nQuant
        z_size = round(z_size)
        R = z_size
        org = Y.ravel()
        means = [None] * nQuant
        images = [None] * (nIter + 1)
        # plt.imshow(imOrig)
        # plt.show()
        images[0] = imOrig
        MSE = [None] * nIter
        for x in range(0, nQuant):
            Z.append(int(R))
            R += z_size
        for j in range(nIter):
            for i in range(nQuant):
                index_0 = Z[i]
                index_1 = Z[i + 1] + 1
                q = np.average(range(index_0, index_1), weights=hist_p[index_0:index_1])
                means[i] = q
            image = Y
            for i in range(nQuant):
                index_0 = Z[i]
                index_1 = Z[i + 1] + 1
                # np.put(image, range(index_0,index_1), means[i])
                image = image.ravel()
                for px in range(0, len(org)):
                    if org[px] >= index_0 and org[px] < index_1:
                        image[px] = means[i]
                image = np.reshape(image, np.shape(Y))

            new_img= yiq_img
            new_img[:,:,0]=image
            # if j==19:
            #     # print(new_img)
            new_img=new_img/255
            new_img = transformYIQ2RGB(new_img)
            # if j==19:
            #     plt.imshow(new_img)
            #     plt.show()
            images[j + 1] = (new_img)
            # MSE[j] = np.square(np.subtract(imOrig, image)).mean()
            MSE[j] = np.square(imgOrig-new_img).mean()

            # print((MSE))
            for i in range(1, nQuant):
                Z[i] = (means[i - 1] + means[i]) / 2
                Z[i] = int(Z[i])
    # print(isGrayScale(image[-1]))
    # plt.imshow(images[19])
    # plt.show()

    # plt.imshow(images[-1], vmin=0, vmax=1)
    # plt.show()
    return images, MSE

    pass
