# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:05:50 2019

@author: Muhammad Ahmed
"""

import numpy as np
import argparse
import imutils
import cv2
import time
import os

# video path to input and output
videopath = "videos/airport.mp4"
outputpath = "output/airport.avi"
# you can also run this program from command line thats why these below functions
#if u want to run this directly from command line please tell me I will make some changes in the code
ap = argparse.ArgumentParser()
ap.add_argument("-y","--yolo", default='yolo-coco')
ap.add_argument("-c","--confidence",type=float,default=0.5)
ap.add_argument("-t","--threshold",type=float,default=0.3)
args = vars(ap.parse_args())
# path to yolo files
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
# generating random colors for labels
COLORS = np.random.randint(0,255,size=(len(LABELS),3),dtype="uint8")
# pre trained yolo weights and configuration path
weightPath = os.path.sep.join([args["yolo"],"yolov3.weights"])
configPath = os.path.sep.join([args["yolo"],"yolov3.cfg"])
#connecting to darknet with the pre trained weights and config
net = cv2.dnn.readNetFromDarknet(configPath,weightPath)
#getting only the output layers
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
#initializing the video stream
vs = cv2.VideoCapture(videopath)
writer = None
(W,H) = (None,None)
# trying to determine the total number of frames in the video
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("Total frames in video {}".format(total))
except:
    print("could not determine")
    total = -1;
#keeping the count of seconds
seconds = 1
while True:
    #reading the next frame from the input file
    (grabbed,frame) = vs.read()
    #it will finish once it reach the end 
    if not grabbed:
        break;
    # if the dimensions are empty, grab them.
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    # constructing the blob from the input file
    blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
    #passing it to the net as input
    net.setInput(blob)
    #keeping track of the time
    start = time.time()
    #getting the output from the input which we provided
    layersOutputs = net.forward(ln)
    #how much time it took
    end = time.time()

    #initializing boxes,confidences and classIDs for each frame
    boxes = []
    confidences = []
    classIDs = []
    #looping over each output layer's output
    for output in layersOutputs:
        #looping over all the detected objects in one output
        for detection in output:
            # keeing the scores,max classID (which have high score) and confidence(probability of that object in seprate variables)
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            #if confidence of detected object is greater than fixed confidence score(we fixed it in the start)
            if confidence > args["confidence"]:
                #gathering the center X,y and width,height of that object
                box = detection[0:4] * np.array([W,H,W,H])
                (centerX,centerY,width,height) = box.astype("int")
                # gathering the x and y axis
                x = int(centerX-(width/2))
                y = int(centerY-(height/2))
                #adding it to the box,confidences and classIDs which we initialized above
                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        #applying the non max suppresion to get rid of duplicate detections based on fixed thresholds        
        idxs = cv2.dnn.NMSBoxes(boxes,confidences,args["confidence"],args["threshold"])
        #if atleast one object is detected        
        if len(idxs) > 0:
            #looping over all the detected objects
            for i in idxs.flatten():
                #gathering the x,y axis and width&height to make the rectangle
                (x,y) = (boxes[i][0], boxes[i][1])
                (w,h) = (boxes[i][2], boxes[i][3])
                #getting the color for that particular object
                color = [int(c) for c in COLORS[classIDs[i]]]
                #drawing the rectangle
                # you can adjust the last parameter to increase the boldness of the box
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,6)
                #putting label with confidence score on rectangle box
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                #you can change the text properties below
                cv2.putText(frame,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,4)
        #updating the writer. it would run only first time. it will write the output file
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(outputpath,fourcc,30,(frame.shape[1],frame.shape[0]),True)
            if total > 0:
                #time taken for single frame
                elap = (end-start)
                #estimated time 
                print("single frame took {} seconds".format(elap))
                print("estimated total time to finish: {}".format(elap * total))
                        
        writer.write(frame)
    print("Seconds: {}".format(seconds))
    seconds = seconds + 1
#releasing the pointers
writer.release()
vs.release()
