# Advanced Lane Finding using Deep Learning
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![Lanes Image](./readme_images/lanelines.gif)

The aim of this project is to develop machine learning model to identify lanelines in a highway driving scenario. We will also calculate radius of curvature and center offset of the vehicle.

The steps used for developing this code are -

1. Compensating for lens distortion
2. Developing image processing pipeline
   * Performing distortion correction for images
   * Applying a combination of different image thresholds for identifying lanelines
   * Isolating region of interest and create a binary image
   * Applying perspective transform on the masked image to obtain a bird's eye view of the images
   * Creating histogram to identify highest pixel concentration
   * Use a sliding windows approach to capture the curvature of laneline
   * Calculate the radius of curvature by fitting a 2nd order polynomial to pixel indices
3. Create a .csv file that stores image information along with the polynomial indices
4. Train the machine learning model using Keras with the .csv file as input
5. Create training and validation sets
6. Test the model using different video stream to check the accuracy of the model

---

## Folder Structure Details

| **File Name** | **Details** |
| :--- | :--- |
| [camera_calculations.ipynb](./camera_calculations.ipynb) | Jupyter Notebook containing pipeline followed for camera calibration |
| [pipeline_images.ipynb](./pipeline_images.ipynb) | Jupyter Notebook containing pipeline followed for identifying images using OpenCV functions
| [video_pipeline.ipynb](./video_pipeline.ipynb) | Code to extract and save image data and lane |parameters |
| [train_cnn.ipynb](./train_cnn.ipynb) | File used for training the model |
| [class_lanelines_1.py](./class_lanelines_1.py) | Python file detecting and storing lanelines information |
| [camera_cal](./camera_cal/) and [camera_cal_outputs](./camera_cal_outputs/) | Directories for original chessboard images and undistorted chessboard images |
| [pickle](./pickle/) | Pickle file storing camera calibration matrices for undistorting images |
| [test_images](./test_images/) and [test_images_output](./test_images_output/) | Directories for test images and thresholded output images |
| [model.h5](./model.h5) - containing a trained convolution neural network |
| [data for lane detection](../data_for_lane_detection/) | This directory is one level up of all our notebook and is used to store input video streams, transformed and thresholded images, neural network training data and some other information |


### Libraries required for running this project -

1. [OpenCV](https://docs.opencv.org/4.4.0/)
2. [NumPy](https://numpy.org/install/)
3. [Natsort](https://github.com/miracle2k/python-glob2)
4. [Glob](https://docs.python.org/3/library/glob.html)
5. [Progressbar](https://progressbar-2.readthedocs.io/en/latest/#install) (for status visualization)

---

## Problem Statement

<span style="font-family:Calibri; font-size:1.5em;">**Using Deep Learning and Convolutional Neural Networks to develop a robust algorithm to identify and draw lanelines in a given video stream.**</span>

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Potential Shortcomings

There are some shortcomings associated with this code.

1. This code and out image threshold is not robust enough to handle the effect of rain/snow, extreme lighting conditions, glare on the camera lens.
2. The current pipeline does not account for the effect of road bumps/potholes, banking, grade on our perspective transform.
3. This code can compensate for undetected lanelines in a couple of frames but is not robust in case of undetected lanelines for several frames continuously.
4. This code works perfectly for highway driving scenario where road curvature is larger but is not well suited for highly twisty roads.

### Improvements

1. To make better predictions in every frame (even in case of different road textures and shadows), a more robust thresholding with different filters (for hsv or rgb channels) can be used.
2. An advanced algorithm to compute the best fit from past several frames.
3. Use of estimation algorithms such as Kalman filters to improve our lane equations
