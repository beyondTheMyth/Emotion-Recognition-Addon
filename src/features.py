import cv2 as cv
import itertools
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def import_dataset():

    project_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
    haar_classifier = cv.CascadeClassifier(os.path.join(project_dir, 'haar/haarcascade_frontalface_alt.xml'))
    dataset_dir = os.path.join(project_dir, 'dataset')
    eye_classifier = cv.CascadeClassifier(os.path.join(project_dir, 'haar/haarcascade_eye.xml'))

    # xtrain = np.array([])
    ytrain = []
    # xtest = np.array([])
    ytest = []
    xtrain = []
    xtest = []
    k = 0
    j = 0
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('jpg'):
                image_path = os.path.join(root, file)
                class_label = os.path.basename(root)
                image_array = cv.imread(image_path)
                image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
                # image_array = cv.GaussianBlur(image_array, (3,5), cv.BORDER_DEFAULT)
                is_training_set = 'train' in os.path.abspath(os.path.join(root, os.pardir))

                # print('image array = {}'.format(image_array))
                # gray_image = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)0
                # 1.02 6
                face = haar_classifier.detectMultiScale(image_array, 1.3, 6)
                    # image_array,
                    # scaleFactor=1.1,
                    # minNeighbors=1,
                    # flags=cv.CASCADE_SCALE_IMAGE
                # )
                for x, y, w, h in face:
                    face_roi = image_array[y:y + h, x: x + w]
                    cropped_face = cv.resize(face_roi, (h, w), interpolation=cv.INTER_AREA)
                    # kernel = np.ones((5, 5), np.float32) / 25
                    # cropped_face = cv.filter2D(cropped_face, -1, kernel)
                    cropped_face = cv.bilateralFilter(cropped_face, 5, 87, 87) #2655
                    cropped_face = cv.GaussianBlur(cropped_face, (3, 7), cv.BORDER_DEFAULT) #3188

                    eyes = eye_classifier.detectMultiScale(cropped_face, 1.3, 3)
                    for ex, ey, ew, eh in eyes:
                        eye_roi = face_roi[ey: ey + eh, ex: ex + ew]
                        cropped_eye = cv.resize(eye_roi, (eh, ew), interpolation=cv.INTER_AREA)
                        feature = cv.Canny(cropped_eye , 60, 180)
                        feature = feature.flatten()
                        if is_training_set:
                            # xtrain = np.vstack(xtrain, feature)
                            xtrain.append(feature)
                            ytrain.append(class_label)

                        else:
                            # xtest = np.vstack(xtest, feature)
                            ytest.append(class_label)
                            xtest.append(feature)
                        k += 1
                        if k == 4:
                            cv.imshow('image', feature)
                            cv.waitKey(100)
                        j += 1
    # xtrain = np.array(xtrain)
    xtrain = np.vstack(xtrain)
    xtest = np.vstack(xtest)
    # xtest = np.array(xtest)
    # xtrain = list(itertools.chain(*xtrain))
    # xtest = list(itertools.chain(*xtest))
    model = SVC(C=1000.0, kernel='poly', gamma='auto')
    model.fit(xtrain, ytrain)
    accuracy = model.score(xtest, ytest)
    print(f'accuracy = {accuracy}')



    print(f'clear dataset size: {j}')