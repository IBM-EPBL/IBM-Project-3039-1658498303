from ast import Import
from cgitb import reset
from flask import Flask,render_template, request
import streamlit as st
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import requests
from keras.models import load_model
from bs4 import BeautifulSoup
import numpy as np

app = Flask(__name__,template_folder='template',static_url_path='/static')
model = load_model('FV.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple','Banana','Bello Pepper','Chilli Pepper','Grapes','Jalepeno','Kiwi','Lemon','Mango','Orange','Paprika','Pear','Pineapple','Pomegranate','Watermelon']
vegetables = ['Beetroot','Cabbage','Capsicum','Carrot','Cauliflower','Corn','Cucumber','Eggplant','Ginger','Lettuce','Onion','Peas','Potato','Raddish','Soy Beans','Spinach','Sweetcorn','Sweetpotato','Tomato','Turnip']

@app.route('/' , methods=['GET'])
def hello_world():
   return render_template('index.html')
@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    img_path="./images/"+imagefile.filename
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    cal = 'https://www.google.com/search?&q=calories in ' + res
    cab = 'https://www.google.com/search?&q=carbs in '+ res
    pro = 'https://www.google.com/search?&q=proteins in '+ res
    fat = 'https://www.google.com/search?&q=fats in '+ res
   
    
    fib = 'https://www.google.com/search?&q=fibre in '+ res
    sodium='https://www.google.com/search?&q=sodium in'+res
    potassium='https://www.google.com/search?&q=potassium in'+res
    cholesterol='https://www.google.com/search?&q=cholesterol in'+res
    req = requests.get(cal).text
    req1 = requests.get(cab).text
    req2 = requests.get(pro).text
    req3= requests.get(fat).text
    req4= requests.get(sodium).text
    req5= requests.get(potassium).text
    req7= requests.get(fat).text
    req6 = requests.get(cholesterol).text
    scrap = BeautifulSoup(req, 'html.parser')
    scrap1=  BeautifulSoup(req1, 'html.parser')
    scrap2=  BeautifulSoup(req2, 'html.parser')
    scrap3=  BeautifulSoup(req3, 'html.parser')
    scrap4=  BeautifulSoup(req4, 'html.parser')
    scrap5=  BeautifulSoup(req5, 'html.parser')
    scrap7=  BeautifulSoup(req7, 'html.parser')
    scrap6=  BeautifulSoup(req6, 'html.parser')
    calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    carbohydrate = scrap1.find("div", class_="BNeawe iBp4i AP7Wnd").text
    protein = scrap2.find("div", class_="BNeawe iBp4i AP7Wnd").text
    fat= scrap3.find("div", class_="BNeawe iBp4i AP7Wnd").text
    sodium=scrap4.find("div", class_="BNeawe iBp4i AP7Wnd").text
    potassium=scrap5.find("div", class_="BNeawe iBp4i AP7Wnd").text
    cholesterol=scrap7.find("div", class_="BNeawe iBp4i AP7Wnd").text
    
    fibre = scrap6.find("div", class_="BNeawe iBp4i AP7Wnd").text
    return render_template('index.html',prediction=res,calories=calories,carbohydrate=carbohydrate,protein=protein,fat=fat,fibre=fibre,cholesterol=cholesterol,sodium=sodium,potassium=potassium)

if __name__ == '__main__':
   app.run(debug = True)