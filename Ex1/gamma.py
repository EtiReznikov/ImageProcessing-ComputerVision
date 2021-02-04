# """
#         '########:'##::::'##::::'##:::
#          ##.....::. ##::'##:::'####:::
#          ##::::::::. ##'##::::.. ##:::
#          ######:::::. ###::::::: ##:::
#          ##...:::::: ## ##:::::: ##:::
#          ##:::::::: ##:. ##::::: ##:::
#          ########: ##:::. ##::'######:
#         ........::..:::::..:::......::
# """

import cv2
import numpy as np
from ex1_utils import imReadAndConvert
from ex1_utils import LOAD_GRAY_SCALE
gamma_slider_max=200
title_window= 'Gamma Correction'
#
img = imReadAndConvert('bac_con.png', 1)

def on_trackbar(val):
    """
    Function to be called every time the slider changes position, and changes the image according to the slider value
    :param val: the slider value
    :return: None
    """
    gamma= val / 100
    corrected_image = np.power(img, gamma)
    cv2.imshow(title_window, corrected_image)



def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    if rep==1:
        img = imReadAndConvert(img_path, rep)
    else:
        img=cv2.imread(img_path).astype(np.float32) / 255
    cv2.namedWindow(title_window, cv2.WINDOW_NORMAL)
    trackbar_name = 'Gamma'
    #create the trackbar
    cv2.createTrackbar(trackbar_name, title_window, 100, gamma_slider_max, on_trackbar)
    # initialize the trackbar with gamma = 1
    on_trackbar(100)
    cv2.waitKey()
    cv2.destroyAllWindows()
    pass


def main():
    gammaDisplay('bac_con.png', 1)

if __name__ == '__main__':
    main()