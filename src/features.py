import cv2 as cv
import numpy as np
import os


def import_dataset():

    project_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
    haar_classifier = cv.CascadeClassifier(os.path.join(project_dir, 'haar/haarcascade_frontalface_alt.xml'))
    dataset_dir = os.path.join(project_dir, 'dataset')
    eye_classifier = cv.CascadeClassifier(os.path.join(project_dir, 'haar/haarcascade_eye.xml'))

    k = 0
    j = 0
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('jpg'):
                image_path = os.path.join(root, file)
                class_label = os.path.basename(root)
                image_array = cv.imread(image_path)
                is_training_set = 'train' in os.path.abspath(os.path.join(root, os.pardir))

                # print('image array = {}'.format(image_array))
                # gray_image = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)0
                # 1.02 6
                face = haar_classifier.detectMultiScale(image_array, 1.1, 6)
                    # image_array,
                    # scaleFactor=1.1,
                    # minNeighbors=1,
                    # flags=cv.CASCADE_SCALE_IMAGE
                # )
                for x, y, w, h in face:
                    face_roi = image_array[y:y + h, x: x + h]
                    cropped_face = cv.resize(face_roi, (w, h), interpolation=cv.INTER_AREA)

                    eyes = eye_classifier.detectMultiScale(cropped_face, 1.05, 3)
                    for ex, ey, ew, eh in eyes:
                        eye_roi = face_roi[ex:ex + ew, ey:ey + eh]
                        cropped_eye = cv.resize(eye_roi, (ew, eh), interpolation=cv.INTER_AREA)

                        k += 1
                        if k == 4:
                            cv.imshow('image', cropped_eye)
                            cv.waitKey(100)
                        if len(face) != 0:
                            j += 1



    print(f'clear dataset size: {j}')