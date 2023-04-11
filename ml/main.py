import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from glob import glob
from sklearn.utils import class_weight
import tensorflow.keras.callbacks as callbacks

train_labels = pd.read_csv("data/train_labels.txt", delimiter=",", header=None, names=["id", "class"], skiprows=1)
val_labels = pd.read_csv("data/validation_labels.txt", delimiter=",", header=None, names=["id", "class"], skiprows=1)

train_labels['id'] = train_labels['id'].apply(lambda x: '{0:0>6}.png'.format(x))
val_labels['id'] = val_labels['id'].apply(lambda x: '{0:0>6}.png'.format(x))


train_labels["class"] = train_labels["class"].astype(str)
val_labels["class"] = val_labels["class"].astype(str)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   horizontal_flip=True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   )

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

test_dir = "data/data/"
test_ids = range(17001, 22150)

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels['class']), y=train_labels['class'])
class_weights_dict = dict(enumerate(class_weights))


train_generator = train_datagen.flow_from_dataframe(
    train_labels,
    directory="data/data/",
    x_col="id",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    #sample_weight=class_weights_dict[train_labels['class'].values]
    )

val_generator = val_datagen.flow_from_dataframe(
    val_labels,
    directory="data/data/",
    x_col="id",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary")

test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({"id": [f"{i:06d}.png" for i in test_ids]}),
    directory=test_dir,
    x_col="id",
    y_col=None,
    target_size=(224, 224),
    batch_size=1,
    class_mode=None,
    shuffle=False)

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples//train_generator.batch_size,
                    epochs=10,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples//val_generator.batch_size,
                    class_weight=class_weights_dict,)


model.evaluate(val_generator)

preds = model.predict(test_generator)
preds = np.where(preds > 0.5, 1, 0)
output_df = pd.DataFrame({'id': range(17001, 22150), 'class': preds.flatten()})
output_df.to_csv("data/submission.csv", sep=',', index=False, header=True)
