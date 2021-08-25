# Programmer Name  : Ms.Ang Pei Jin, Asia Pacific University Student
# Program Name     : Social Distancing Detector
# Description      : Detect physical distance between two individuals
# First written on : Monday, 24 May 2021
# Edited on        : Sunday, 30 May 2021

# base path to YOLO directory
MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections along with the
# threshold when applying non-maxim suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# should NVIDIA CUDA GPU be used?
USE_GPU = True

# define the minimum safe distance (in cm) that two people can be from each other
MIN_DISTANCE = 100

