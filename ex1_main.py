from ex1_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
import time


def histEqDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    # hsitogramEqualize(img)
    imgeq, histOrg, histEq = hsitogramEqualize(img)

    # Display cumsum
    cumsum = np.cumsum(histOrg)
    cumsumEq = np.cumsum(histEq)
    plt.gray()
    plt.plot(range(256), cumsum, 'r')
    plt.plot(range(256), cumsumEq, 'g')

    # Display the images
    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(imgeq)
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()
    #
    img_lst, err_lst = quantizeImage(img, 3, 20)
    # quantizeImage(img, 3, 20)
    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %f" % err_lst[0])
    print("Error last:\t %f" % err_lst[-1])


    O=imReadAndConvert('quant_RGB_q4_it5.png', LOAD_RGB )
    plt.imshow(O)
    plt.show()
    O=O*255

    print(O)

    print((img_lst[-1])*255)

    plt.gray()
    plt.imshow(img_lst[0])
    plt.figure()
    plt.imshow(img_lst[-1])

    plt.figure()
    plt.plot(err_lst, 'r')
    plt.show()


def main():
    print("ID:", myID())
    img_path = 'beach.jpg'

    # Basic read and display
    # imDisplay(img_path, LOAD_GRAY_SCALE)
    # imDisplay(img_path, LOAD_RGB)

    # Convert Color spaces
    # img = imReadAndConvert(img_path, LOAD_RGB)


    #
    # n= imReadAndConvert('histEQ_RGB.png', LOAD_RGB)
    # yiq_img = transformRGB2YIQ(n)
    # yiq_img=yiq_img*255
    # print(yiq_img[:,:,0])
    # # n=n*255
    #
    # print(yiq_img[:,:,0])


    # yiq_img = transformRGB2YIQ(img)
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(img)
    # ax[1].imshow(yiq_img)
    # plt.show()
    #
    # # Image histEq
    # histEqDemo(img_path, LOAD_GRAY_SCALE)
    # histEqDemo(img_path, LOAD_RGB)
    # # #
    # # Image Quantization
    # quantDemo(img_path, LOAD_GRAY_SCALE)
    quantDemo(img_path, LOAD_RGB)
    #
    # # Gamma
    # gammaDisplay(img_path, LOAD_GRAY_SCALE)
    # gammaDisplay(img_path, LOAD_RGB)

if __name__ == '__main__':
    main()
