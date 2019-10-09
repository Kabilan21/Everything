import tensorflow as tf
from tensorflow import keras
import numpy as np

import cv2
model = keras.models.load_model("D:\\CNN1.tf")

face_cascade = cv2.CascadeClassifier('C:\\Users\\Kabilan\\Downloads\\haar\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        roi_gray = gray[y:y + h, x:x + w]
        z = cv2.resize(roi_gray, (200,200))
        z = np.array(z).reshape(-1, 200,200, 1)
        z = z / 255.0
        L = model.predict(z)
        print(L)
    cv2.imshow('img',frame)
    cv2.waitKey(1)

