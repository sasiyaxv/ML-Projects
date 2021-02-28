# -------------------------------------------------------------------------------------

# A sample model for the famous human or horse dataset
# 2/28/2021

# -------------------------------------------------------------------------------------



from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.models import model_from_json
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import numpy as np
import random
import pickle
import keras
import cv2
import os

# path and lables of the dataset
DATADIRECTORY = "Dataset"
CATEGORIES = ["horses","humans"]

# looping through the data and converting all images to grayscale because it is easy to train 
for category in CATEGORIES:
	path = os.path.join(DATADIRECTORY,category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
		plt.imshow(img_array, cmap = "gray")
		break
	break


#if want resize the images
IMG_SIZE = 50

# resize the image
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))

# creating the training data
training_data = []

def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIRECTORY,category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
				training_data.append([new_array,class_num])
# if the dataset has corrupted images
			except Exception as e:
				pass

create_training_data()
print(len(training_data))

#impportant to shuffle the data to make the neural network learn effectively
random.shuffle(training_data)	


x = []
y = []

for features, label in training_data:
	x.append(features)
	y.append(label)

x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)

# --------------------way of serializing obj of arr-----------------------------
pickle_out = open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()
# --------------------------------------------------------------------------------

# ----------------------reloading ---------------------------------
# pickle_in = open("x.pickle","rb")
# x = pickle.load(pickle_in)
# --------------------------------------------------------------------------------

# converting to np arrays
x=np.array(x/255.0)
y=np.array(y)

# creating the model
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#dropout layer used to ..if CNN tries to remeber the data to drop some connections
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#since 2 outputs this loss function wil work
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
# ,callbacks=[cp_callback]

model.fit(x, y, batch_size=32, epochs=3, validation_split=0.1)

model.save('saveData.model')

# ------------------------------------------------------------------------save model ----------------------------

# serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)


# directory = "SavedModel"
# os.mkdir(directory)
# model.save_weights("SavedModel/model.h5")
# print("Saved model to disk")


# ------------------------------------------------------------------------save model ----------------------------






			


