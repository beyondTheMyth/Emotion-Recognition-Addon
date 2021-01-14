from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(8, 2)
# data = []

project_dir = os.path.abspath(os.path.join(__file__, os.pardir))
dataset_dir = os.path.join(project_dir, '../dataset')
validation_dir = os.path.join(dataset_dir, 'validation')

haar_classifier = cv2.CascadeClassifier(os.path.join(project_dir, '../haar/haarcascade_frontalface_alt.xml'))

k = 0
j = 0

# Load training hist and labels
with open("training_data.csv", "rb") as file:
	# read the file to numpy array
	data = np.load(file)

with open("training_labels.csv", "rb") as file:
	# read the file to numpy array
	labels = np.load(file)


# Train a Linear SVM on the data
model = SVC(C=10, kernel='rbf', random_state=42, gamma='scale')
model.fit(data, labels)

vlabels = []

# loop over the validation images
for root, dirs, files in os.walk(validation_dir):
	for file in files:
		if file.endswith('jpg'):
			image_path = os.path.join(root, file)
			class_label = os.path.basename(root)
			image_array = plt.imread(image_path)
			vlabels.append(class_label)

			face = haar_classifier.detectMultiScale(image_array, 1.05, 6)
			for x, y, w, h in face:
				face_roi = image_array[y:y + h, x: x + w]
				cropped_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
				cropped_face = cv2.GaussianBlur(cropped_face, (3, 7), cv2.BORDER_DEFAULT)  # 3188
				hist = desc.describe(cropped_face)
				prediction = model.predict(hist.reshape(1, -1))

				if class_label == prediction[0]:
					j += 1
				k += 1

print(k, j)
print(j/k*100)
