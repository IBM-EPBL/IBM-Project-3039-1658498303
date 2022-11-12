
from flask import Flask,render_template,request
import os

import numpy as np

import requests
from tensorflow.keras.models import load_model #to load our trained model
from tensorflow.keras.preprocessing import image
import requests

app = Flask(__name__,template_folder='template') #initializing a flask app 
# Loading the model
model=load_model('nutrition.h5')
print("Loaded model from disk")

@ app.route('/')# route to display the home page
def home():
    print("Loaded model from disk")
    return render_template('Home.html') #rendering the home page

@ app.route('/image1', methods=['GET', 'POST']) # routes to the index html
def image1():
    return render_template("Image.html")


@ app.route('/predict' ,methods=['GET','POST']) # route to show the predictions in a Web UI
def lanuch():
    if request.method=='POST':
        f=request.files['file'] # requesting the file
        f
        basepath=os.path.dirname('__file__') #storing the file directory
        print(basepath)
        filepath=os.path.join(basepath,"test",f.filename)
        
         #storing the file in uploads folder
        f.save(filepath) #saving the file
        
        img=image.load_img(filepath,target_size=(32,32)) #load and reshaping the image
        x=image.img_to_array(img) #converting image to an array
        x=np.expand_dims(x,axis=0) #changing the dimensions of the image
        
        pred=np.argmax(model.predict(x), axis=1)
        print("prediction",pred) #printing the prediction
        index=['APPLE','BANANA','ORANGE','BANANA','WATERMELON','WATERMELON','WATERMELON','APPLE','BANANA','WATERMELON']
        
        result=str(index[pred[0]])
        print(result)
        x=result
        result=nutrition(result)
        print(result)
        
        return render_template("0.html",showcase=(result),showcase1=(x))
def nutrition(index):
    
    import requests

    url = "https://calorieninjas.p.rapidapi.com/v1/nutrition"

    querystring = {"query":index}

    headers = {
	"X-RapidAPI-Key": "85887549f4msh51e7315b280a87ep1f43e0jsn585c940f2ea6",
	"X-RapidAPI-Host": "calorieninjas.p.rapidapi.com"
     }

    response = requests.request("GET", url, headers=headers, params=querystring)

    print(response.text)
    return response.json()['items']
if __name__ == "__main__":
    # running the app
    app.run(debug=False) 
       