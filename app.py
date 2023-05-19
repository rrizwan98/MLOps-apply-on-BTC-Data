from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        pred_day=request.form.get('pred_day')

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_day)
        return render_template('home.html',results=results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")   