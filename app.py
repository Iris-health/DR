from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re

import numpy as numpy
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn as nn
#import cv2 as cv
from torchvision import models, transforms
import PIL
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)



# Model saved with Keras model.save()
MODEL_PATH = './2catg_incptnv3_model.pt'

# Load your trained model
mdl = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
mdl.eval()
#md=models.inception_v3(pretrained=True)
#md.fc=nn.Sequential(nn.Linear(2048,500),nn.ReLU(),nn.Linear(500,50),nn.ReLU(),nn.Linear(50,2))

def model_predict(img_path):
    im = Image.open(img_path)
    im = np.array(im)
    #im = cv.resize(im,(448,448))
    #im = cv.cvtColor(np.array(im), cv.COLOR_BGR2RGB)
    #im=cv.addWeighted ( im,4, cv.GaussianBlur( im , (0,0) , 8) ,-4 ,140)
    im=np.array(im)
    im=im/255.0
    im=torch.Tensor(im)
    im=torch.unsqueeze(im,0)
    im=im.permute(0,3,1,2)


    


    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!

    preds = mdl(im)
    return preds



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        preds = model_predict(f)
        result='NAN'
        if(preds[0][0]>preds[0][1]):
            result='Normal'
        else:
            result='Diabetic'

            
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

