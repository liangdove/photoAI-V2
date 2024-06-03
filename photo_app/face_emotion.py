import dlib
import cv2
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.views.decorators import gzip

# 表情识别算法，基于Dlib

class face_emotion:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("C:\\E\\Django_learn\\login\\shape_predictor_68_face_landmarks.dat")

    def detect_emotion(self, frame):
        # Convert frame to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        rects = self.detector(img_gray, 0)

        for d in rects:
            shape = self.predictor(frame, d)
            mouth_width = (shape.part(54).x - shape.part(48).x) / (d.right() - d.left())
            mouth_height = (shape.part(66).y - shape.part(62).y) / (d.right() - d.left())

            brow_sum = 0
            frown_sum = 0
            line_brow_x = []
            line_brow_y = []
            for j in range(17, 21):
                brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
                frown_sum += shape.part(j + 5).x - shape.part(j).x
                line_brow_x.append(shape.part(j).x)
                line_brow_y.append(shape.part(j).y)

            tempx = np.array(line_brow_x)
            tempy = np.array(line_brow_y)
            z1 = np.polyfit(tempx, tempy, 1)
            brow_k = -round(z1[0], 3)

            brow_height = (brow_sum / 10) / (d.right() - d.left())
            brow_width = (frown_sum / 5) / (d.right() - d.left())

            eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                       shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
            eye_height = (eye_sum / 4) / (d.right() - d.left())

            if round(mouth_height >= 0.03):
                if eye_height >= 0.056:
                    emotion = "amazing"
                else:
                    emotion = "happy"
            else:
                if brow_k <= -0.3:
                    emotion = "angry"
                else:
                    emotion = "nature"

            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 4)
        return frame