# Programmer Name  : Ms.Ang Pei Jin, Asia Pacific University Student
# Program Name     : Social Distancing Detector
# Description      : Detect physical distance between two individuals
# First written on : Monday, 24 May 2021
# Edited on        : Sunday, 30 May 2021
from configs import config
from configs.detection import detect_people
from scipy.spatial import distance as dist
from influxdb import InfluxDBClient
import numpy as np
import queue, threading, time
import argparse
import imutils
import cv2
import os
import serial

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels the YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
# weights - model - neural network data for recognise people
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load the YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if GPU is to be used or not
if config.USE_GPU:
    # set CUDA s the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the "output" layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
# open input video if available else webcam stream
# vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# connecting to the influxdb
client = InfluxDBClient('localhost', 8086)
client.switch_database('scdb')

#invoke Arduino
ser = serial.Serial('COM4', 9600);

# original coordinate of table in image
original1 = (622, 590)
original2 = (947, 455)
original3 = (1066, 541)
original4 = (754, 699)

# table width and height in cm
real_length = 127
real_width = 69.4
pixel_length = 449
pixel_width = 245

# transformed width and height in pixel
# transformed1 = (264, 259)
# transformed2 = (transformed1[0] + pixel_length, transformed1[1])
# transformed3 = (transformed2[0], transformed1[1] + pixel_width)
# transformed4 = (transformed1[0], transformed3[1])

pts = np.array([original1, original2, original3, original4], dtype=np.float32)
ipm_pts = np.array([[622,190], [1071,190], [1071,435], [622,435]], dtype=np.float32)
ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        # create new thread for reading the video frame process
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


# cap = VideoCapture(0)
# while True:
#     time.sleep(.5)  # simulate time between events
#     frame = vs.read()
#     cv2.imshow("frame", frame)
#     if chr(cv2.waitKey(1) & 255) == 'q':
#         break

vs = VideoCapture(args["input"] if args["input"] else 0)

# loop over the frames from the video stream
while True:
    # vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    # read the next frame from the input video

    # (grabbed, frame) = vs.read()
    # main request the frame from the helper thread
    frame = vs.read()
    # grabbed,frame = vs.read()

    # if the frame was not grabbed, then that's the end fo the stream
    # if not grabbed:
    #     break

    # resize the frame and then detect people (only people) in it
    # frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    # initialize the set of indexes that violate the minimum social distance
    violate = set()

    # ensure there are at least two people detections (required in order to compute the
    # the pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        print(results)
        centroids = np.array([[[r[2][0], r[1][3]] for r in results]], dtype=np.float32)
        # transformed centroids
        # centroids = np.array([[r[2] for r in results]], dtype=np.float32)
        tc = cv2.perspectiveTransform(centroids, ipm_matrix)
        print (tc)
        D = dist.cdist(tc[0], tc[0], metric="euclidean")
        print (D)

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE * pixel_width / real_width:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then update the color and send to Arduino
        if i in violate:
            color = (0, 0, 255)
            ser.write('s'.encode())

        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on the output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    if len(violate) > 0:
        ser.write('s'.encode())

    # check to see if the output frame should be displayed to the screen
    if args["display"] > 0:
        frame2 = imutils.resize(frame, width=900)
        # show the output frame
        cv2.imshow("Output", frame2)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break

    # if an output video file path has been supplied and the video writer ahs not been 
    # initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        print("[INFO] writing stream to output")
        writer.write(frame)

    # overwrite the data into the influxdb
    json_body = [
        {
            "measurement": "public_details",
            "tags": {
                "place": "home",
            },
            "fields": {
                "number_of_people": len(results),
                "violate": len(violate),
            }
        }]

    client.write_points(json_body)
