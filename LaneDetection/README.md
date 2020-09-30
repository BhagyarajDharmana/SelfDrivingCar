# **Project 1: Finding Lane Lines on the Road** 

## Introduction

For a self driving vehicle to stay in a lane, the first step is to identify lane lines before issuing commands to the control system. Since the lane lines can be of different colors (white, yellow) or forms (solid, dashed) this seemingly trivial task becomes increasingly difficult. Moreover, the situation is further exacerbated with variations in lighting conditions. Thankfully, there are a number of mathematical tools and approaches available nowadays to effectively extract lane lines from an image or dashcam video. In this project, a program is written in python to identify lane lines on the road, first in an image, and later in a video stream. After a thorough discussion of the methodology, potential shortcomings and future improvements are suggested.

## Methodology
Before attempting to detect lane lines in a video, a software pipeline is developed for lane detection in a series of images. Only after ensuring that it works satisfactorily for test images, the pipeline is employed for lane detection in a video. 
The pipeline consisted of 5 major steps excluding reading and writing the image. 

Consider the test image given below:

![Figure 1](https://github.com/BhagyarajDharmana/SelfDrivingCar/tree/master/LaneDetection/writeup_images/solidWhiteRight.jpg)

The test image is first converted to grayscale from RGB using the helper function grayscale(). This produces the below image.

![Figure 2](https://github.com/BhagyarajDharmana/SelfDrivingCar/tree/master/LaneDetection/writeup_images/_gray_solidWhiteRight.jpg)

The grayscaled image is given a gaussian blur to remove noise or spurious gradients. The blurred image is given below.

![Figure 3](https://github.com/BhagyarajDharmana/SelfDrivingCar/tree/master/LaneDetection/writeup_images/_blur_gray_solidWhiteRight.jpg)

Canny edge detection is applied on this blurred image and a binary image shown below is produced.

![Figure 4](https://github.com/BhagyarajDharmana/SelfDrivingCar/tree/master/LaneDetection/writeup_images/_edges_solidWhiteRight.jpg)

This image contains edges that are not relevant for lane finding problem. A region of interest is defined to separate the lanes from sorrounding environment and a masked image containing only the lanes is extracted using cv2.bitwise_and() function from opencv library. This can be seen below.

![Figure 5](https://github.com/BhagyarajDharmana/SelfDrivingCar/tree/master/LaneDetection/writeup_imagess/_masked_edges_solidWhiteRight.jpg)

This binary image of identified lane lines is finally merged with the original image using cv2.addweighted() function from opencv library. This produces an image given below. Note that, this is without making any modifications to the drawlines() helper function. It can be observed that the lines are not continuous as required.

![Figure 6](https://github.com/BhagyarajDharmana/SelfDrivingCar/tree/master/LaneDetection/writeup_images/_lines_edges_solidWhiteRight.jpg)
