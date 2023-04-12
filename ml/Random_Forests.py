import numpy as np
import pandas as pd
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

train_labels = pd.read_csv('data/train_labels.txt', delimiter=',', header=None, names=['id', 'class'], skiprows=1)
val_labels = pd.read_csv('data/validation_labels.txt', delimiter=',', header=None, names=['id', 'class'], skiprows=1)

train_images = []
val_images = []
for i in range(1, 15001):
    img = cv2.imread(f"data/data/{i:06}.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    train_images.append(img.flatten())

for i in range(15001, 17001):
    img = cv2.imread(f"data/data/{i:06}.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    val_images.append(img.flatten())

train_images = np.array(train_images)
val_images = np.array(val_images)

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
clf.fit(train_images, train_labels['class'])

val_pred = clf.predict(val_images)
val_f1 = f1_score(val_labels['class'], val_pred, average='macro')
print(f"Validation f1 score: {val_f1}")

test_images = []
for i in range(17001, 22150):
    img = cv2.imread(f"data/data/{i:06}.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    test_images.append(img.flatten())


test_images = np.array(test_images)
test_pred = clf.predict(test_images)

submission_df = pd.DataFrame({'id': range(17001, 22150), 'class': test_pred})
submission_df.to_csv('data/submission.csv', sep=",", index=False, header=True)

