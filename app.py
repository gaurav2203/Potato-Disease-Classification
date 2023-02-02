import numpy as np
import requests
from io import BytesIO
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app= Flask(__name__)

endpoint= 'http://localhost:8501/v1/models/potato_disease_model:predict'
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.route('/')
def hello_world():
    return "Hello World!"

@app.route('/home')
def home():
    return render_template('temp.html')

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif' , 'JPG'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict(filename):
    img= tf.keras.utils.load_img(filename, target_size= (256, 256))
    img_array= tf.keras.utils.img_to_array(img)
    img_array_reshape= img_array.reshape(1 , 256 , 256,3)
    payload= {
                "instances": img_array_reshape.tolist()
            }
    # Making POST request
    r= requests.post(endpoint, json= payload)
    prediction = np.array(r.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

@app.route('/predict', methods= ['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        if not request.files['file']:
            return jsonify({"status": 400, "message": "No Image is Passed"})
        img= BytesIO(request.files['file'].read())
        predicted_class, confidence= predict(img)
        return render_template('temp.html', predicted_class= predicted_class, confidence= confidence)
    else:
        return render_template('temp.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
