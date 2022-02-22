import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model import RecommendClass

app = Flask(__name__)
recom_model = RecommendClass()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_name = [str(x) for x in request.form.values()][0]
    output = recom_model.recommend_products(user_name.lower())
    return render_template('index.html', prediction_label="Top 5 Recommended Products for the user : '{}' are listed below -".format(user_name.lower()), 
            prediction_text0='a.) '+output[0],
            prediction_text1='b.) '+output[1],
            prediction_text2='c.) '+output[2],
            prediction_text3='d.) '+output[3],
            prediction_text4='e.) '+output[4])

if __name__=='__main__':
    app.run(debug=True)