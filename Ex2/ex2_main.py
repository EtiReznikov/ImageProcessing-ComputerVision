from ex2_utils import *
import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt
import time


#helper functions to plot all the resuluts
def convolution_test_2D(img:np.ndarray,kernel:np.ndarray):
    cv2_solution= cv2.filter2D(img, -1, cv2.flip(kernel,0), borderType=cv2.BORDER_REPLICATE)
    ny_solution = conv2D(img, kernel)
    f, ax = plt.subplots(1, 2)
    plt.gray()
    f.suptitle("2D convolution", fontsize=30)
    ax[0].imshow(cv2_solution)
    ax[1].imshow(ny_solution)
    ax[0].set_title('filter2D', size=20)
    ax[1].set_title('conv2D', size=20)
    plt.show()

def derivatives_test(img:np.ndarray):
    directions, magnitude, d_x, d_y = convDerivative(img)

    magnitude = np.float_(magnitude)
    magnitude *= 255.0 / np.max(magnitude)

    directions = directions.astype('float32')
    directions = (directions - np.min(directions)) / np.max(directions)

    f, axarr = plt.subplots(1, 5, figsize=(20, 20))
    plt.gray()
    f.suptitle("Image derivatives", fontsize=40)
    axarr[0].imshow(img)
    axarr[0].set_title('original', size=20)
    axarr[1].imshow(d_x, cmap='gray')
    axarr[1].set_title('X Derivation', size=20)
    axarr[2].imshow(d_y)
    axarr[2].set_title('Y Derivation', size=20)
    axarr[3].imshow(magnitude)
    axarr[3].set_title('magnitude', size=20)
    axarr[4].imshow(directions)
    axarr[4].set_title('directions', size=20)
    f.suptitle('Derivatives', fontsize=16)
    plt.show()

def blur_test(img:np.ndarray, kernel_size:np.ndarray):
    blur1=blurImage1(img, kernel_size)
    blur2=blurImage2(img, kernel_size)

    f, axarr = plt.subplots(1, 2, figsize=(20, 20))
    plt.gray()
    f.suptitle("Bluring", fontsize=30)
    axarr[0].imshow(blur1)
    axarr[0].set_title('blur1', size=20)
    axarr[1].imshow(blur2)
    axarr[1].set_title('blur2', size=20)
    plt.show()

def  edgeDetectionSobel_test(img:np.ndarray, thresh: float = 0.7):

    sobel_result, my_result = edgeDetectionSobel(img, thresh)
    f, axarr = plt.subplots(1, 2, figsize=(20, 20))
    plt.gray()
    f.suptitle("Sobel edge detection", fontsize=30)
    axarr[0].imshow(sobel_result)
    axarr[0].set_title('cv2 result', size=20)
    axarr[1].imshow(my_result)
    axarr[1].set_title("my result, thresh:"+str(thresh), size=20)
    plt.show()

def edgeDetection_zerocrossing_test(image:np.ndarray):

    ans_simple=edgeDetectionZeroCrossingSimple(image)
    ans_LOG = edgeDetectionZeroCrossingLOG(image)

    f, axarr = plt.subplots(1, 2, figsize=(20, 20))
    plt.gray()
    f.suptitle("Zerro crossing edge detection", fontsize=30)
    axarr[0].imshow(ans_simple)
    axarr[0].set_title('Zero crossing simple', size=20)
    axarr[1].imshow(ans_LOG)
    axarr[1].set_title('Zero crossing LOG', size=20)
    plt.show()
    ### add here test to zero crossing

def edgeDetectionCanny_test(img: np.ndarray, thrs_1: float, thrs_2: float):
    cv2_a, ans = edgeDetectionCanny(img, thrs_1, thrs_2)
    f, axarr = plt.subplots(1, 2, figsize=(20, 20))
    plt.gray()
    f.suptitle("Canny edge detection", fontsize=30)
    axarr[0].imshow(ans)
    axarr[0].set_title('my result', size=20)
    axarr[1].imshow(cv2_a)
    axarr[1].set_title('cv2 result', size=20)
    plt.show()


def main():
    print("ID:", myID())

    #1D convolution

    #Testing the 1D convolution function using 100 random iterations and compare it to the numpy convolution function results
    correct_answers=0
    for i in range(0,100):
        input_size= random.randrange(5,11)
        input=np.random.rand(input_size)
        kernel_size= random.randrange(1,6)
        kernel=np.random.rand(kernel_size)
        ans=conv1D(input,kernel)
        np_ans=np.convolve(input,kernel, 'full')
        if (np.allclose(ans, np_ans)):
            correct_answers=correct_answers+1
    print('correct answers of 1D convolution: ' , correct_answers, 'out of 100')

    # 2D convolution
    img_path = 'Rectangles.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    k_size = 15
    kernel = np.ones((k_size, k_size))
    kernel /= kernel.sum()
    convolution_test_2D(img, kernel)

    img_path = 'coins.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    k_size = 15
    kernel = np.ones((k_size, k_size))
    kernel /= kernel.sum()
    convolution_test_2D(img, kernel)

    #Derivatives
    derivatives_test(img)
    img_path = 'Rectangles.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    derivatives_test(img)

    #Blur
    image = cv2.imread('69-2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size=3
    blur_test(image, kernel_size)

    #edgeDetectionSobel
    image = cv2.imread('Lenna.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edgeDetectionSobel_test(image)

    image = cv2.imread('zebra.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edgeDetectionSobel_test(image, 0.5)


    #edgeDetection_ZeroCrossing and edgeDetectionLOG
    image_path='Lenna.png'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edgeDetection_zerocrossing_test(image)
    #the results with LOG are a little better

    #edgeDetectionCanny
    image_path='zebra.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edgeDetectionCanny_test(image,50,100)


    #houghCircle
    canvas= np.ones((250,250),  dtype=np.uint8)*255
    cv2.circle(canvas, (100,100),10, (0,0,0), -1)
    cv2.circle(canvas, (150, 150), 12,  (0,0,0), -1)
    plt.imshow(canvas)
    plt.show()
    st = time.time()
    circle=houghCircle(canvas, 5, 25)
    print("Time:%.2f" % (time.time() - st))
    print(circle)



if __name__ == '__main__':
    main()