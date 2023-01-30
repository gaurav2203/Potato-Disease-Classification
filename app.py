import os
import uuid
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array
import numpy as np
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


app= Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model= load_model('./saved_models/1')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"] 

@app.route("/", methods=['GET'])
def hello():
    return "Hello World"

@app.route("/home")
def home():
    return render_template('temp.html')

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif' , 'JPG'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict(filename, model):
    img= load_img(filename, target_size= (256, 256))
    img_array= img_to_array(img)
    img_array_reshape= img_array.reshape(1 , 256 , 256,3)
    prediction= model.predict(img_array_reshape)
    print(prediction.tolist())
    index= np.argmax(prediction[0])
    predicted_class= CLASS_NAMES[index]
    confidence= (np.max(prediction[0])*100)
    confidence= round(confidence, 2)
    return predicted_class, confidence

@app.route('/predict', methods=['GET', 'POST'])
def predict_output():
    error= ''
    target_img_path= os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if (request.files):
            file= request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img_path, file.filename))
                img_path= os.path.join(target_img_path, file.filename)
                img= file.filename
                # return img_path
                print(img_path)
                predicted_class, confidence= predict(img_path, model)
            else:
                error= "Please upload images of jpg, jpeg and png extension only"

            if(len(error) == 0):
                return render_template('temp.html', predicted_class= predicted_class, confidence= confidence)
            else:
                return error
    else:
        return render_template('temp.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port= 8000)