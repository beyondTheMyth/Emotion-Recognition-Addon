import cv2 as cv
import numpy as np
import os


def import_dataset():

    project_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
    haar_classifier = cv.CascadeClassifier(os.path.join(project_dir, 'haar/haarcascade_frontalface_alt2.xml'))
    dataset_dir = os.path.join(project_dir, 'dataset')

    k = 0
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('jpg'):
                image_path = os.path.join(root, file)
                class_label = os.path.basename(root)
                image_array = cv.imread(image_path)
                # print('image array = {}'.format(image_array))
                # gray_image = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
                eyes = haar_classifier.detectMultiScale(image_array, 1.3, 5)
                    # image_array,
                    # scaleFactor=1.1,
                    # minNeighbors=1,
                    # flags=cv.CASCADE_SCALE_IMAGE
                # )
                print(f'eyes are: {eyes}')
                k += 1
                if k == 4:
                    cv.imshow('image', image_array)
                    cv.waitKey(100)


