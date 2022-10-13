
from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
from flasgger import Swagger
import pickle

from flask import Flask, request

app = Flask(__name__)
swagger = Swagger(app)

model = load_model("./ImageRecog.h5")

@app.route("/predict_digit", methods = ['POST'])

def predict_digit():
    """Example endpoint returning a prediction of mnist
    ---
    parameters:
        - name: image
          in: formData
          type: file
          required: true
    """
    im = Image.open(request.files['image'])
    im2arr = np.array(im).reshape((1,  28, 28, 1))
    prediction = str(np.argmax(model.predict([im2arr,im2arr])))
    print(f'Selected Image is {prediction}')
    return prediction

if __name__ == '__main__':
    app.run()


# path = "C:\\Users\\Rajesh.Pandey\\PycharmProjects\\ImageAPI\\mnist.npz"
# (X_train, y_train), (X_test, y_test)= tf.keras.datasets.mnist.load_data(path)
#
# for i in np.random.randint(0,10000+1, 10):
#     arr2im = Image.fromarray(X_train[i])
#     arr2im.save('{}.png'.format(i),"PNG")






