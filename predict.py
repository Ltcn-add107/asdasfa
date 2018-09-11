import cv2
import tensorflow as tf
import keras
CATEGORIES = ["traffic", "road"]


def prepare(filepath):
    IMG_SIZE = 80  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("traffic_detection.model")
model.summary()
prediction = model.predict([prepare('traffic1.jpg')])
print(prediction)  # will be a list in a list.
if (prediction[0][0]>prediction[0][1]):
    print ('traffic')
else:
    print('road')
