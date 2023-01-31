import json, os, requests
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from tensorflow.keras.preprocessing import image
import numpy as np

app= Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
endpoint= "http://localhost:8501/v1/models/potato_disease_model:predict"
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

# def predict(filename):
#     img= image.load_img(filename, target_size= (256, 256))
#     img_array= image.img_to_array(img)
#     img_array_reshape= img_array.reshape(1 , 256 , 256, 3)

#     # img_final= img.astype('float16')
#     payload= {
#         "instances": [{'input_image': img_array_reshape.tolist()}]
#     }

#     # Sending POST request to tf serving server
#     response= requests.post(endpoint, data=json.dumps(payload))
#     prediction= np.array(response.json()["predictions"][0])

#     index= np.argmax(prediction[0])
#     predicted_class= CLASS_NAMES[index]
#     confidence= (np.max(prediction[0])*100)
#     confidence= round(confidence, 2)
#     return predicted_class, confidence

@app.route('/predict', methods=['GET', 'POST'])
def predict_output():
    error= ''
    data={}
    target_img_path= os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if (request.files):
            file= request.files['file']
            if file and allowed_file(file.filename):
                img= image.img_to_array(image.load_img(BytesIO(request.files["file"].read()), target_size=(256, 256)))/ 255.
                img= img.astype('float16')
                payload= {
                    "instances": [{'input_image': img.tolist()}]
                }
                r= requests.post(endpoint, json= payload)
                prediction= json.loads(r.content.decode('utf-8'))

                index= np.argmax(prediction[0])
                predicted_class= CLASS_NAMES[index]
                confidence= (np.max(prediction[0])*100)
                confidence= round(confidence, 2)
                return predicted_class, confidence

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