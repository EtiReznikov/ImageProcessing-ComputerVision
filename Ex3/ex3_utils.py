import sys
from typing import List

import numpy as np
import cv2
import math
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

gauusian_kernel = cv2.getGaussianKernel(5, 0.3)
def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[y,x]...], [[dU,dV]...] for each points
    """
    w= win_size//2
    im1 = cv2.GaussianBlur(im1 , (3, 3) , 0 )
    im2 = cv2.GaussianBlur(im2, (3, 3), 0)
    k_x = np.array([[1, 0, -1]])
    k_y = np.array([[1], [0], [-1]])
    I_x = cv2.filter2D(im2, -1, k_x)
    I_y = cv2.filter2D(im2, -1, k_y)
    I_t = im2-im1

    dU_dV=[]
    x_y=[]
    for i in range(w, im1.shape[0] - w, step_size):
        for j in range(w, im2.shape[1] - w, step_size):
            Ix = I_x[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = I_y[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = I_t[i - w:i + w + 1, j - w:j + w + 1].flatten()

            b = np.reshape(It, (It.shape[0], 1))  # get b
            A = np.vstack((Ix, Iy)).T  # get A
            A_t=np.transpose(A) # get A^t
            A_tA= np.matmul(A_t, A)
            flag= satisfy_eigenvalues(A_tA)
            if flag:
                nu = np.dot(np.linalg.pinv(A), b)
                temp_x_y= np.array([j,i])
                dU_dV.append(nu)
                x_y.append(temp_x_y)

    x_y= np.array(x_y)
    x_y=np.reshape(x_y,(-1,2))

    dU_dV= np.array(dU_dV)
    dU_dV=np.reshape(dU_dV,(-1,2))

    return x_y, dU_dV
    pass

def satisfy_eigenvalues(AtA: np.ndarray)-> bool:
    """
    Given a mat, returns if its eigenvalues are satisfy according  to LK
    :param AtA: mat
    :return: if its eigenvalues are satisfy
    """
    w,v= np.linalg.eig(AtA)
    G1= w[0]
    G2= w[1]
    if (G1<G2):
        temp=G2
        G2=G1
    if G2>1 and G1/G2<100:
      return True

    return False

def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    img= crop_img(img,levels)
    gaussian_pyramid= gaussianPyr(img, levels)
    laplaceian_pyramid=[]
    filter=gauusian_kernel
    for i in range(len(gaussian_pyramid)-1):
        expanded= gaussExpand(gaussian_pyramid[i+1], filter)
        temp = gaussian_pyramid[i] - expanded
        laplaceian_pyramid.append(temp)
    laplaceian_pyramid.append(gaussian_pyramid[levels - 1])
    return laplaceian_pyramid
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    img= lap_pyr[-1]
    filter=gauusian_kernel
    for i in range(len(lap_pyr) -1 ,0, -1):
        img = gaussExpand(img, filter) + lap_pyr[i-1]
    return img
    pass


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    temp_img=crop_img(img,levels)
    pyramid= [temp_img]
    filter=gauusian_kernel
    for i in range (0, levels-1):
        temp_img= cv2.filter2D(temp_img, -1, filter)
        temp_img= temp_img[::2 , ::2]
        pyramid.append((temp_img))
    return pyramid
    pass


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    shape= np.shape(img)
    height, width = shape[0], shape[1]
    if (len(shape)==3):
        expand_img= np.zeros((height*2, width*2,3))
    else:
        expand_img = np.zeros((height * 2, width * 2))

    expand_img[::2,::2]=img
    expand_img=cv2.filter2D(expand_img, -1, 4*gs_k)
    return expand_img
    pass


def crop_img(img: np.ndarray, levels: int)-> np.ndarray:
    """
    crop image for x levels to 2^x[img_size/ 2^x]
    :param img: Image
    :param levels: x
    :return: the croped image
    """
    two_pow_x= 2**levels
    shape=np.shape(img)
    rows=shape[0]
    cols=shape[1]
    rows= two_pow_x*(math.floor(rows/two_pow_x))
    cols= two_pow_x*(math.floor(cols/two_pow_x))
    img=img[0:rows, 0:cols]
    return img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """

    img_1= crop_img(img_1,levels)
    img_2= crop_img(img_2,levels)
    mask=crop_img(mask, levels)

    mask_gaussian_pyr = gaussianPyr(mask, levels)
    naive_blend= np.zeros(np.shape(img_1))

    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if mask[i][j][0]==0:
                naive_blend[i][j]=img_2[i][j]
            else:
                naive_blend[i][j]=img_1[i][j]

    img_1_laplaceian_pyr= laplaceianReduce(img_1, levels)
    img_2_laplaceian_pyr= laplaceianReduce(img_2, levels)

    laplaceian= []
    for i in range(len(img_1_laplaceian_pyr)):
        laplaceian.append(mask_gaussian_pyr[i]*img_1_laplaceian_pyr[i]+(1-mask_gaussian_pyr[i])*img_2_laplaceian_pyr[i])

    return naive_blend, np.clip(laplaceianExpand(laplaceian), 0, 1)

    pass
