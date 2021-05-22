import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import sys 
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)


IMG_SIZE = (160, 160)


data_augmentation = tf.keras.Sequential(
    [tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SIZE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE,include_top=False,weights='imagenet')
base_model.trainable = False
base_model.summary()
 
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  

prediction_layer = tf.keras.layers.Dense(1)  

inputs = tf.keras.Input(shape = (160,160,3))
# x = data_augmentation(inputs)
x = preprocess_input(inputs)
x = base_model(x,training = False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs,outputs)


checkpoint_path = "training_1/cp.ckpt"
model.load_weights(checkpoint_path)  
  
 
def predict(image_path):
    images = image.load_img(image_path, target_size=(160,160))
    images = image.img_to_array(images)
    images = np.expand_dims(images, axis=0)
    predictions  = model.predict(images)
    return predictions,classes[1 if predictions[0][0]>0 else 0]

classes=['NORMAL','PNEUMONIA']
# image_path = 'C:/Users/SIBSANKAR/Desktop/pneumonia/data/train/NORMAL/IM-0115-0001.jpeg'
# image_path = 'C:/Users/SIBSANKAR/Desktop/pneumonia/data/train/PNEUMONIA/person1_bacteria_1.jpeg'

if len(sys.argv)>1:
    image_path = sys.argv[1]
    _,c = predict(image_path)
    print(c)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg' }
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/', methods=['GET', 'POST'])
def pneumonia_detect():
    print(request.method)
    print(request.files)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No File')
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('No Selected file')
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print('Got a file!!!')
            uid = str(uuid.uuid4())
            filename = secure_filename(uid+"_"+file.filename)
            new_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(new_path)
            _,pred = predict(new_path)
            print('Pred',pred)
            return pred
        
    return 'File key should be \'file\''

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    