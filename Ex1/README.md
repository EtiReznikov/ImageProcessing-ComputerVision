# image-processing

First assignment at Image Processing and Computer Vision course.
I used Python 3 to write and test the program.

##functions

imReadAndConvert- Reads an image, and returns normalize numpy array according to the requested representation.

imDisplay-Reads an image and displays it according to the requested representation.

transformRGB2YIQ- Converts an RGB image to YIQ color space.

transformYIQ2RGB- Converts an YIQ image to RGB color space.

hsitogramEqualize- Equalizes the histogram of an image.

isGrayScale- Identify if grayscale or color img loaded.

getYIQdata- Returns YIQ image information data- YIQimg, Y chanel, histogram of y chanle and the bins of the histogram.

quantizeImage- Quantized an image in to n colors.

gammaDisplay- Gamma correction gui.

on_trackbar- Function to be called every time the slider changes position, and changes the image according to the slider value.

##test images

testImg1- I used it to test the histogram equation on an image with a gray sky

testImg2- I used it to test image quantization on a multi-color image.
