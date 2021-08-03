import numpy as np
import cv2
stopcascade=cv2.CascadeClassifier("stopsign_classifier.xml")
vid=cv2.VideoCapture(0)
while True:
    ret,img=vid.read()
    imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sign=stopcascade.detectMultiScale(imggray,1.1,4)
    for (x,y,w,h) in sign:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    cv2.imshow("output",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break