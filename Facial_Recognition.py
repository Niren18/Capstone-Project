import sys
import sklearn
from kivy.uix.screenmanager import ScreenManager, Screen
import math
from kivy.clock import Clock
from kivy.graphics import Color, Line
import os
import time
import cv2
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
import numpy as np
import face_recognition
from datetime import datetime
import datetime
import pandas as pd
from imutils.video import VideoStream
from keras.utils import img_to_array
from keras.models import load_model
import pickle
import imutils
import threading
prototxt = './deploy.prototxt'
caffemodel = './res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
model = load_model('liveness.model')
le = pickle.loads(open('le.pickle', "rb").read())

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



class FaceEncodingInputScreen():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.image = Image()
        self.success_dialog = None

    def on_enter(self, *args):
        self.ids.FIS_but.bind(on_press=self.callback)
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update, 1.0 / 15.0)
        return self.image

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.video_FIS.texture = texture


class LoginScreen():

    def load_images(self):
        path = './Face_Directory'
        self.classNames = []
        self.encodeListKnown = []
        images = []
        myList = os.listdir(path)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
        self.encodeListKnown = findEncodings(images)

    def on_enter(self, *args):
        global Encoding_Reload_Prompt
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update, 1.0 / 15.0)
        Clock.schedule_interval(self.Go_Home,20)
        if self.Encoding_Reload_Prompt == True:
            self.load_images()
            self.Encoding_Reload_Prompt = False
        return self.image

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.video.texture = texture
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float") / 255.0  
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    preds = model.predict(face)[0]
                    j = np.argmax(preds)
                    label = le.classes_[j]
                    if label == "real":
                        facesCurFrame = face_recognition.face_locations(frame)
                        encodesCurFrame = face_recognition.face_encodings(frame, facesCurFrame)
                        rects = net.forward()
                        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                            matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace, tolerance=0.4)
                            faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                            matchIndex = np.argmin(faceDis)

                            if matches[matchIndex]:
                                name = self.classNames[matchIndex].upper()
                                y1, x2, y2, x1 = faceLoc
                                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                                            1, cv2.LINE_AA)
                                cv2.putText(frame, "[INFORMATION]: PLEASE CLICK ON PROCEED BUTTON TO CONTINUE:",
                                            (1, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                                label = "{}: {:.4f}".format(label, preds[j])
                                cv2.putText(frame, name, (startX, startY - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.rectangle(frame, (startX, startY), (endX, endY),
                                              (0, 255, 0), 2)                                
                                global name_db
                                name_db = name
                    

   