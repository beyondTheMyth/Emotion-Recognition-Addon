# import the necessary packages
import numpy as np
from localbinarypatterns import LocalBinaryPatterns
import pyautogui
import imutils
import time
import os
import cv2
import pickle
import json

class EmotionRecognitionApp:
    def __init__(self, config_path):
        self.__config_dict = self.__load_config(config_path)
        self.__desc = LocalBinaryPatterns(8, 2)
        self.__model = None
        self.__export_path = self.__config_dict['export_path']
        try:
            with open(self.__config_dict['model_path'], 'rb') as model_file:
                self.__model = pickle.load(model_file)
        except IOError as ioe:
            print(f"Error: Couldn't open file {self.__config_dict['model_path']}")
            print(ioe)
        except OSError as ose:
            print(f"Error: Couldn't open file {self.__config_dict['model_path']}")
            print(ose)
        self.__project_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
        self.__haar_classifier = cv2.CascadeClassifier(os.path.join(self.__project_dir, 'haar/haarcascade_frontalface_alt.xml'))
        pyautogui.screenshot()

    def run(self, frames_per_emotion, debug=False):
        x0, y0, x1, y1 = self.__config_dict['frame_coordinates']
        width = x1 - x0
        height = y1 - y0
        for i in range(999999):
            mean_prediction = ''
            predictions = []
            for j in range(frames_per_emotion):
                image = pyautogui.screenshot(region=(x0, y0, width, height))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                face = self.__haar_classifier.detectMultiScale(image, 1.03, 6)
                for x, y, w, h in face:
                    face_roi = image[y:y + h, x: x + w]
                    cropped_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
                    cv2.imshow(f'image{i}', cropped_face)
                    cropped_face = cv2.GaussianBlur(cropped_face, (3, 7), cv2.BORDER_DEFAULT)  # 3188
                    hist = self.__desc.describe(cropped_face)
                    prediction = self.__model.predict_proba(hist.reshape(1, -1))
                    print(prediction[0])
                    predictions.append(self.__get_emotion(list(prediction[0])))

            if len(predictions) > 0:
                mean_prediction = self.__choose_prediction(predictions)
                mean_prediction = self.__index_to_emotion(mean_prediction)
                print(f'{mean_prediction} {i}')
                self.__export_emotion(mean_prediction, i)

    def __load_config(self, config_path):
        try:
            with open(config_path, 'rb') as config_file:
                config = json.load(config_file)
                return config
        except IOError as ioe:
            print(ioe)
        except OSError as ose:
            print(ose)
        return None

    def __get_emotion(self, emotions_list):
        max_conf = max(emotions_list)
        most_relevant = emotions_list.index(max_conf)
        if max_conf > 0.50:
            return most_relevant, max_conf
        else:
            return 1, emotions_list[1]

    def __choose_prediction(self, predictions):
        negs = 0
        pos = 0
        neut = 0
        for most_relevant, max_conf in predictions:
            if most_relevant == 0:
                negs += 1 + 1 * max_conf
            elif most_relevant == 1:
                neut += 1 + 1 * max_conf
            elif most_relevant == 2:
                pos += 1.1 + 1 * max_conf
            else:
                print('Error: no such emotion')

        emotion_sums = (negs, neut, pos)
        maximum_sum = max(emotion_sums)
        return emotion_sums.index(maximum_sum)

    def __index_to_emotion(self, index):
        if index == 0:
            return 'negative'
        elif index == 1:
            return 'neutral'
        elif index == 2:
            return 'positive'
        else:
            return 'invalid_emotion'

    def __export_emotion(self, emotion, emotion_id):
        try:
            with open(self.__export_path, "w") as exfile:
                exfile.write(f'{emotion} {emotion_id}')
        except IOError as ioe:
            print(ioe)
        except OSError as ose:
            print(ose)


