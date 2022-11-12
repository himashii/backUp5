# import libraries
from flask import Flask, request
from flask_cors import CORS
import json
import werkzeug
from blockchain.Blockchain import Blockchain
import blockchain.Block as b
from face_age_gender_rec.face_recognition import face_rec
import os
from face_age_gender_rec.age_gender_recognition import ager_rec
from blockchain.model import User
import cv2
from keras.models import Sequential, save_model, load_model
import numpy as np
import joblib
import librosa
from ocr import easy_ocr
from ocr import tesseract_ocr
from voice_rec import google_cloud_speech_rec_english as v_rec_eng
import subprocess

app = Flask(__name__)
CORS(app)

blockchain_obj = Blockchain()
model = load_model('ocr/cnn/model.h5')

receipt_detect_type = 'cnn'  # or ocr


@app.route('/login', methods=['GET', 'POST'])
def login():
    imagefile = request.files['image']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('upload/' + filename)

    face_list = os.listdir('blockchain/face_images')
    face_rec_status = False
    rec_id = 0
    for i in face_list:
        if face_rec.verifyFace(i, filename):
            face_rec_status = True
            rec_id = i.split(".")[0]

    user_type = blockchain_obj.get_user_type_by_id(rec_id)

    if face_rec_status:
        age, gender = ager_rec.get_gender_and_age('upload/' + filename)
        return_str = '[{"status" : 1, "age" : ' + str(age) + ', "gender" : "' + str(
            gender) + '", "user_level" : "' + user_type + '"}]'
    else:
        return_str = '[{"status" : 0, "age" : 0, "gender" : "male", "user_level" : "Null"}]'

    return json.loads(return_str)


@app.route('/manual_login', methods=['GET', 'POST'])
def manual_login():
    email = request.form['email']
    password = request.form['password']

    blockchain_login_status, image_path = blockchain_obj.login(email, password)

    if blockchain_login_status:
        age, gender = ager_rec.get_gender_and_age(image_path)
        return_str = '[{"status" : 1, "age" : ' + str(age) + ', "gender" : "' + str(
            gender) + '"}]'
    else:
        return_str = '[{"status": 0, "age": 0, "gender" : "male"}]'

    return json.loads(return_str)


@app.route('/register', methods=['GET', 'POST'])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    user_level = request.form['user_level']
    imagefile = request.files['image']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('upload/' + filename)

    img = cv2.imread('upload/' + filename)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

    if len(faces_rect) > 0:
        save_file_name = 'blockchain/face_images/' + str(len(blockchain_obj.get_chain())) + '.jpg'
        cv2.imwrite(save_file_name, img)
        user = User(name, email, password, user_level, save_file_name)
        blockchain_obj.mine(b.Block(user))
        age, gender = ager_rec.get_gender_and_age('upload/' + filename)
        return_str = '[{ "status" : 1}]'
    else:
        return_str = '[{"status" : 0 }]'
    return json.loads(return_str)


# @app.route('/receipt_reader', methods=['GET', 'POST'])
# def receipt_reader():
#     imagefile = request.files['image']
#
#     filename = werkzeug.utils.secure_filename(imagefile.filename)
#     print("\nReceived image File name : " + imagefile.filename)
#     imagefile.save('upload/' + filename)
#
#     if receipt_detect_type == 'cnn':
#         img = cv2.imread('upload/' + filename)
#         # img = cv2.imread('Test/' + name)
#         img = np.asarray(img)
#         img = cv2.resize(img, (32, 32))
#         img = preProcessing(img)
#         img = img.reshape(1, 32, 32, 1)
#         classIndex = int(model.predict_classes(img))
#         predictions = model.predict(img)
#         # probVal = np.amax(predictions)
#         a = np.amax(predictions) * 100
#         x = "%.2f" % round(a, 2)
#         med_name_dict = {0: "Allermine", 1: "Amitone", 2: "Amoxicillin", 3: "Astifen", 4: "candozol", 5: "Ciptolet",
#                          6: "Dacterine", 7: "telday", 8: "Ventoline", 9: "Zithrin"}
#         return_str = '[{ "medicine name" : "' + str(med_name_dict[classIndex]) + '" }]'
#     else:
#         text = easy_ocr.get_ocr(filename)
#         # text = tesseract_ocr.get_ocr(filename) # using tesseract
#         return_str = '[{ "medicine name" : "' + str(text) + '" }]'
#
#     return json.loads(return_str)

@app.route('/receipt_reader', methods=['GET', 'POST'])
def receipt_reader():
    imagefile = request.files['image']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('upload/' + filename)

    if receipt_detect_type == 'cnn':
        img = cv2.imread('upload/' + filename)
        # img = cv2.imread('Test/' + name)
        img = np.asarray(img)
        img = cv2.resize(img, (32, 32))
        img = preProcessing(img)
        img = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img)
        probVal = np.amax(predictions)
        print(probVal)
        final_pred = np.argmax(predictions, axis=1)
        print('----------------------')
        a = np.amax(predictions) * 100
        x = "%.2f" % round(a, 2)
        med_name_dict = {0: "Allermine", 1: "Amitone", 2: "Amoxicillin", 3: "Astifen", 4: "candozol", 5: "Ciptolet",
                         6: "Dacterine", 7: "telday", 8: "Ventoline", 9: "Zithrin"}
        return_str = '[{ "medicine name" : "' + str(med_name_dict[final_pred[0]]) + '" }]'
    else:
        text = easy_ocr.get_ocr(filename)
        # text = tesseract_ocr.get_ocr(filename) # using tesseract
        return_str = '[{ "medicine name" : "' + str(text) + '" }]'

    return json.loads(return_str)


# @app.route('/voice_reader', methods=['GET', 'POST'])
# def voice_reader():
#     imagefile = request.files['voice']
#
#     filename = werkzeug.utils.secure_filename(imagefile.filename)
#     print("\nReceived voice File name : " + imagefile.filename)
#     imagefile.save('audios/' + filename)
#
#     res = v_rec_eng.get_name_from_voice(filename)
#
#     return_str = '[{ "medicine name" : "' + str(res) + '"}]'
#     return json.loads(return_str)


@app.route('/voice_classification', methods=['GET', 'POST'])
def voice_classification():
    audio_file = request.files['audio']
    filename = werkzeug.utils.secure_filename(audio_file.filename)
    print("\nReceived audio File name : " + audio_file.filename)
    audio_file.save('audios/' + filename)

    if filename.endswith(".3gp"):
        command = 'ffmpeg -i audios/' + filename + ' audios/' + filename.replace('.3gp', '.wav') + ' -y'
        subprocess.call(command, shell=True)
        #print('true')

    #print(filename)

    loaded_rf = joblib.load("voice_rec/random_forest.joblib")
    mfccs, rms, spectral_flux, zcr = feature_extraction('audios/' + str(filename.replace('.3gp', '.wav')))
    extracted_features = np.hstack([mfccs, rms, spectral_flux, zcr])

    y_prediction = loaded_rf.predict([extracted_features])

    out_dict = {0: 'amlodipine', 1: 'benadryl', 2: 'benzonatate', 3: 'diphenhydramin', 4: 'hydromet', 5: 'lisinopril',
                6: 'losartan', 7: 'metaprolol', 8: 'panadol', 9: 'thyronorm'}

    return_str = '[{ "medicine name" : "' + str(out_dict[y_prediction[0]]) + '" }]'

    return json.loads(return_str)


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def feature_extraction(file_name):
    # X, sample_rate = sf.read(file_name, dtype='float32')
    X, sample_rate = librosa.load(file_name, sr=None)  # Can also load file using librosa
    if X.ndim > 1:
        X = X[:, 0]
    X = X.T

    stft = np.abs(librosa.stft(X))

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0)  # Returns N_mel coefs
    rms = np.mean(librosa.feature.rms(y=X).T, axis=0)  # RMS Energy for each Frame (Stanford's). Returns 1 value
    spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T,
                            axis=0)  # Spectral Flux (Stanford's). Returns 1 Value
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)  # Returns 1 value

    return mfccs, rms, spectral_flux, zcr


if __name__ == "__main__":
    app.run(host="192.168.43.77", port=5000, debug=True)
