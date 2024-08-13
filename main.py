import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime



timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


rails_dir = "trainingdata/rails"
stairs_dir = "trainingdata/stairs"

#load images to a single path with assigned label
def load_images_from_directory(directory, label, img_size=(224, 224)):
    images = []
    labels = []
    for img in os.listdir(directory):
        img_path = os.path.join(directory, img)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, img_size)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

#load data from the rails path with label 0
rails_images, rails_labels = load_images_from_directory(rails_dir, label=0)

#load data from the stairs path with label 1
stairs_images, stairs_labels = load_images_from_directory(stairs_dir, label=1)

#combine the data together
train_images = np.concatenate((rails_images, stairs_images), axis=0)
train_labels = np.concatenate((rails_labels, stairs_labels), axis=0)


train_images = train_images / 200.0

#split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

#9 layer CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)), #Finds edges with 3x3 pixels
    tf.keras.layers.MaxPooling2D(2, 2), #reduces image size ( faster processing )
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),#2D -> 1D
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(2, activation='softmax')  #2 Output (ex. rail or stairs)
])

#compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

#eval model
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc}")

model.save(f'skate_obstacle_classifier_{timenow}.h5')
