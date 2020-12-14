import cv2 as cv
import numpy as np
import os


def import_dataset():
    project_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
    dataset_dir = os.path.join(project_dir, 'dataset')

    k = 0
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('jpg'):
                image_path = os.path.join(root, file)
                class_label = os.path.basename(root)
                image_array = cv.imread(image_path)
                # print('image array = {}'.format(image_array))
                gray_image = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
                k += 1
                if k == 4:
                    cv.imshow('image', gray_image)
                    cv.waitKey(100)


