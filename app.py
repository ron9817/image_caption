from flask import Flask, render_template, request
import datetime


import numpy as np
from numpy import array
import pandas as pd
#import matplotlib.pyplot as plt

import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from sklearn.externals import joblib
import pickle


app=Flask(__name__)
@app.route("/")
def index():
  return render_template("index.html")
@app.route("/img", methods=["POST"])
def caption():
  if request.method=="POST":
    datime=str(datetime.datetime.now())
    f=request.files['photo']
    #save_url='./static/img'+datime+'.jpg'
    save_url='./static/img.jpg'
    f.save(save_url)
    ixtoword="i2w.pkl"
    wordtoix="w2i.pkl"
    weights="model_30.h5"
    #img_name='./static/img'+datime+'.jpg'
    img_name='./static/img.jpg'
    c=Img_caption(ixtoword, wordtoix, weights, img_name)
    caption=c.predict()
    #return render_template("caption.html", img_src='./static/img'+datime+'.jpg', caption=caption)
    return render_template("caption.html", img_src='./static/img.jpg', caption=caption)
  
  else:
    return("<h1>Go to home page and upload the image to get caption</h1>")
  
  
class Img_caption:
  def __init__(self, ixtoword, wordtoix, weights, img_name):
    self.ixtoword=joblib.load(ixtoword)
    self.wordtoix=joblib.load(wordtoix)
    self.weights=weights
    self.max_length=34
    self.vocab_size=1652
    self.embedding_dim=200
    model_img = InceptionV3(weights='imagenet')
    self.model_img_new = Model(model_img.input, model_img.layers[-2].output)
    self.img_name=img_name
    
  def predict(self):
    self.extract_features(self.img_name)
    
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(self.max_length,))
    se1 = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
    self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    self.model.compile(loss='categorical_crossentropy', optimizer='adam')
    self.model.load_weights(self.weights)
    with open("encoded_test_images.pkl", "rb") as encoded_pickle:
        encoding_test = pickle.load(encoded_pickle)
    images = ''
    pic = self.img_name
    image = encoding_test.reshape((1,2048))
    #x=plt.imread(images+pic)
    #plt.imshow(x)
    #plt.show()
    #print("Greedy:",self.greedySearch(image))
    caption=self.greedySearch(image)
    return caption
    
    
  def greedySearch(self, photo):
    in_text = 'startseq'
    for i in range(self.max_length):
        sequence = [self.wordtoix[w] for w in in_text.split() if w in self.wordtoix]
        sequence = pad_sequences([sequence], maxlen=self.max_length)
        yhat = self.model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = self.ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
  
  def preprocess(self, image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
  
  def encode(self,image):
    image = self.preprocess(image)
    fea_vec = self.model_img_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec
  
  
  def extract_features(self, img):
    encoding_test = self.encode(img)
    with open("encoded_test_images.pkl", "wb") as encoded_pickle:
        pickle.dump(encoding_test, encoded_pickle)
        
if __name__=="__main__":
  app.run()
          
