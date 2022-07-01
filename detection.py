

from recognition import *
import cv2
import numpy as np

net = cv2.dnn.readNet('yolo.weights','yolo.cfg')

classes = []
with open("classes.names", "r") as f:
    classes = f.read().splitlines()
    #video input
cap = cv2.VideoCapture('tes3.mp4')
font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
               
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            #extraction
            if (x != 0) and (y != 0) and (w != 0) and (h != 0):
                if confidences[i]> 0.999: 
                    if (w > 30 and h > 20):  
                        roi = img[y:y+h,x:x+w]
                        cv2.imwrite('img.png',roi)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(img, label + " " + confidence, (x, y-1), font, 1, (255,255,255), 2)
    
    '''cv2.namedWindow("test", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("test", 1200, 800) '''
    cv2.imshow('test',img)
    key = cv2.waitKey(1)
    if key==27:
        break
im = cv2.imread('img.png')
recog(im)
cap.release()
cv2.destroyAllWindows()
