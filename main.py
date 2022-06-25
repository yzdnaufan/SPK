from flask import Flask, render_template,request
import pickle 
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor

model = pickle.load(open('Emisi_XGB_model.pkl', 'rb'))
arr = [[2.4 , 12, 0, 18]]
df = pd.DataFrame(arr)

print(model.predict(df))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [[float(x) for x in request.form.values()]]
    final_features = pd.DataFrame(int_features)
    prediction = model.predict(final_features)
    output = round(prediction[0], 2) 
    
    if output > 200 :
        return render_template('index.html', prediction_text='CO2    Emission of the vehicle is {} '.format(output), saran = 'Kendaraan anda disarankan melakukan pemeriksaan mesin')
    else :
        return render_template('index.html', prediction_text='CO2    Emission of the vehicle is {}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)

