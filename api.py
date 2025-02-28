from flask import Flask, render_template, request
import json
from io import BytesIO
import base64
from models import  load_and_preprocess_image, make_prediction
api = Flask(__name__)



@api.route('/', methods=["GET"])
def home():
    return render_template("index.html")


@api.route('/predict', methods = ["POST"])
def prediction_endpoint():
    image = request.files['image-input']
    image_bytes = BytesIO()
    image.save(image_bytes)
    predicted_class = make_prediction(load_and_preprocess_image(image_bytes=image_bytes.getvalue()))
    return json.dumps(predicted_class)




if __name__ == "__main__":
    api.run(debug=True)