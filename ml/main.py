import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from glob import glob

train_labels = pd.read_csv("data/train_labels.txt", delimiter=",", header=None, names=["id", "class"], skiprows=1)
val_labels = pd.read_csv("data/validation_labels.txt", delimiter=",", header=None, names=["id", "class"], skiprows=1)

train_labels['id'] = train_labels['id'].apply(lambda x: '{0:0>6}.png'.format(x))
val_labels['id'] = val_labels['id'].apply(lambda x: '{0:0>6}.png'.format(x))


train_labels["class"] = train_labels["class"].astype(str)
val_labels["class"] = val_labels["class"].astype(str)


# Create image data generators for the training, validation, and test sets
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

test_dir = "data/data/"
test_ids = range(17001, 22150)

train_generator = train_datagen.flow_from_dataframe(
    train_labels,
    directory="data/data/",
    x_col="id",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary")

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

# Define the CNN architecture
# Define the CNN architecture
model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(scale=1./255),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples//train_generator.batch_size,
                    epochs=1,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples//val_generator.batch_size)

# Evaluate the model on the validation set
model.evaluate(val_generator)

# Make predictions on the test set and output the results in the desired format
preds = model.predict(test_generator)
preds = np.where(preds > 0.5, 1, 0)
output_df = pd.DataFrame({'id': range(17001, 22150), 'class': preds.flatten()})
output_df.to_csv("data/submission.csv", sep=',', index=False, header=False)
