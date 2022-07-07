
# Imports required for flask and its relevant functions
from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer

#tensorflow import for processing
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Memory allocation
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = InteractiveSession(config=config)

# numpy for encoding predictions
import numpy as np
# util python file for image loading
from util import base64_to_pil

# Declare that its a flask application
app = Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH = 'models/customResnet_small.h5'


# Load the trained model
model = load_model(MODEL_PATH)
model._make_predict_function()
print('Model loaded. Start serving...')

# Prediction function

def model_predict(img, model):
    img = img.resize((224, 224))

    # Pre-processing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        # pred_proba = "{:.3f}".format(np.amax(preds))  # Max probability
        pred_class = np.argmax(preds, axis=1)

        result = str(pred_class)  # Convert to string

        print(preds)
        print(result)
        # print(pred_proba)

        if result == '[0]':
            prediction = "Baroque"

        if result == '[1]':
            prediction = "Expressionism"

        if result == '[2]':
            prediction = "Post Impressionism"

        if result == '[3]':
            prediction = "Romanticism"

        if result == '[4]':
            prediction = "Symbolism"

        print(prediction)

    # Below is the code to get the second class

        preds[np.where(preds==np.max(preds))] = 0

        pred_class = np.argmax(preds, axis=1)

        result = str(pred_class)  # Convert to string

        print(preds)
        print(result)

        if result == '[0]':
            second_prediction = "Baroque"

        if result == '[1]':
            second_prediction = "Expressionism"

        if result == '[2]':
            second_prediction = "Post Impressionism"

        if result == '[3]':
            second_prediction = "Romanticism"

        if result == '[4]':
            second_prediction = "Symbolism"

        print(second_prediction)


        # Serialize the result
        return jsonify(result=prediction, probability=second_prediction)

    return None


if __name__ == '__main__':

    # Serve created using gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

