from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
import numpy as np
from flask import Flask, render_template, request, send_from_directory
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

model = load_model("static/model.h5", compile=False)

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (299,299))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 299, 299, 3)
    pred = model.predict(img_arr)
    dpreds = decode_predictions(pred, top=5)[0]
    print(dpreds)
    COUNT = COUNT + 1
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_arr[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
    exp_image = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imsave("exp images/exp.jpg", exp_image)
    return render_template('prediction.html', data=dpreds)

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory("exp images", "exp.jpg")

if __name__ == '__main__':
    app.run()

    
