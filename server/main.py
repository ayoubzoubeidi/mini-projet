import cv2
import os
import tensorflow as tf
import numpy as np
from flask import Flask, request
from flask_cors import CORS
from flask import render_template
from tensorflow.keras.models import load_model

#Labeling function required for load_learner to work
def GetLabel(fileName):
  return fileName.split('-')[0]
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
model = load_model("my_model.h5") #Import Model
app = Flask(__name__)
cors = CORS(app) #Request will get blocked otherwise on Localhost

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    f = request.files['file']
    img_path = "static" + f.filename
    f.save(img_path)    
    img = cv2.imread(img_path)
    resize=cv2.resize(img, dsize = (128,128))
    prediction = model.predict(np.expand_dims(resize, 0))
    namecode = np.argmax(prediction , axis=-1)
    
    if namecode == 0:
        type = "Benign"
    elif namecode == 1:
        type = "Malignant"
    elif namecode == 2: 
        type = "Normal"
    score = prediction[0][2]
    print(prediction)
    return f'{type} with a precision of {score*100:.1f}% {prediction}'

if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000)


