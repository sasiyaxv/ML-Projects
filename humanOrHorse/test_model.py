# script to reload the saved model
import cv2
import tensorflow as tf

CATEGORIES = ["horse", "human"]  


def prepare(filepath):
    IMG_SIZE = 200  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  


model = tf.keras.models.load_model("saveData.model")

prediction = model.predict([prepare('22.JPG')]) 
print(prediction)

print(CATEGORIES[int(prediction[0][0])])


# references - https://pythonprogramming.net/using-trained-model-deep-learning-python-tensorflow-keras/?completed=/tensorboard-optimizing-models-deep-learning-python-tensorflow-keras/