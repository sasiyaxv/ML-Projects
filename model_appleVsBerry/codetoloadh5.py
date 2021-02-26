import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pydot
import matplotlib.pyplot as plt

image_size = (180, 180)
batch_size = 32


# Recreate the exact same model
new_model = tf.keras.models.load_model('save_at_25.h5')

# Show the model 
new_model.summary()

#provide image path
img = keras.preprocessing.image.load_img(
    "Download.JPG", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  

predictions = new_model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent apple and %.2f percent other."
    % (100 * (1 - score), 100 * score)
)