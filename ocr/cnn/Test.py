from tensorflow.keras.models import Sequential, save_model, load_model
import numpy as np
import cv2

width = 640
height = 480
threshold = 0.65
cameraNo = 0


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def test(name):
    model = load_model('model.h5')
    img = cv2.imread('Test/' + name)
    # img = cv2.imread('Test/' + name)
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    # probVal = np.amax(predictions)
    a = np.amax(predictions) * 100
    x = "%.2f" % round(a, 2)
    return classIndex


print(test('1.jpg'))