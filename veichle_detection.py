# -*- coding: utf-8 -*-
"""Veichle Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_T0XwsnuWY7zE1vrsfurYJmlak9hGbbR
"""

!pip install opencv-python-headless
!pip install opencv-python

image = cv2.imread('/content/cars.jpg')
image_arr = np.array(image)
image_arr = np.asarray(image_arr, dtype=np.uint8)

from PIL import Image
import cv2
import numpy as np
import requests
from google.colab.patches import cv2_imshow


# Downloading and resizing the image from the URL
# image_url = 'https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg'

# # Load an image from a URL
# url = image_url
# response = requests.get(url)
# image_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
# image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
cv2_imshow(image_arr)

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2_imshow(grey)

blur = cv2.GaussianBlur(grey, (5, 5), 0)
cv2_imshow(blur)

dilated = cv2.dilate(blur, np.ones((3, 3)))
cv2_imshow(dilated)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
cv2_imshow(closing)

car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)

cnt = 0
for (x, y, w, h) in cars:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cnt += 1

# Print the total number of detected cars and buses
print(cnt, " cars found")
# Convert the annotated image to PIL Image format and display it
annotated_image = Image.fromarray(image)
annotated_image.show()

cv2_imshow(image)





"""#02"""

!pip install opencv-contrib-python==4.5.4.60 --force-reinstall

# prompt: I want to read file from Google drive

from google.colab import drive
drive.mount('/content/drive')

import os
drive.mount('/content/drive')
yolo_folder_path = '/content/drive/My Drive/yolo'  # Replace with the actual path to your yolo folder
for filename in os.listdir(yolo_folder_path):
    print("/content/drive/My Drive/yolo" + filename)



import numpy as np
import imutils
import cv2

inputVideoPath = '/content/drive/My Drive/yolo/854671-hd_1920_1080_25fps.mp4'
outputVideoPath = 'OJ_video.avi'
yoloWeightsPath = '/content/drive/My Drive/yolo/yolov4-csp.weights'
yoloConfigPath = '/content/drive/My Drive/yolo/yolov4-csp.cfg'
detectionProbabilityThresh = 0.5
nonMaximaSuppression = 0.3

labelsPath = '/content/drive/My Drive/yolo/coco_class.txt'
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(yoloConfigPath, yoloWeightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(inputVideoPath)
writer = None
(W, H) = (None,None)

try:
    prop = cv2.CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))

except:
    total = -1
    print('Frames could not be determined')

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None and H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > detectionProbabilityThresh:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, detectionProbabilityThresh, nonMaximaSuppression)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(outputVideoPath, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

# prompt: I want to watch .avi video here

from IPython.display import HTML
from base64 import b64encode
mp4 = open('OJ_video.avi','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)





import cv2

# Define input and output video paths
input_video_path = "/content/drive/MyDrive/OJ_video.avi"
output_video_path = "output_video.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()

from moviepy.editor import *
inputVideoPathTwo = '/content/drive/MyDrive/OJ_video.avi'
outputVideoPathTwo = 'vid_1_output.gif'


clipTwo = (VideoFileClip(inputVideoPathTwo).subclip((0.0),(20.0)).resize(0.2))
clipTwo.write_gif(outputVideoPathTwo,fps=15)

from moviepy.editor import *

# Path to the input video
inputVideoPathTwo = '/content/drive/MyDrive/OJ_video.avi'
# Path to the output video
outputVideoPathTwo = 'vid_1_output.mp4'

# Load the video, trim it to 20 seconds, resize it, and save it as an MP4 file
clipTwo = VideoFileClip(inputVideoPathTwo).subclip(0, 10).resize(0.2)
clipTwo.write_videofile(outputVideoPathTwo, fps=15)

