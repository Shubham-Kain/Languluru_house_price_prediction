# from flask import Flask,render_template
#
# app =  Flask(__name__)
#
# @app.route('/')
# def index():
#     return render_template('index.html')
import pickle
import pandas as pd
import numpy as np
import math

import streamlit as st
from nltk.app.nemo_app import colors
from sqlalchemy import column
from streamlit import columns

model = pickle.load(open("XGBRegressor_pipe.pkl",'rb'))

data = pd.read_csv("Clean_Bengaluru_price.csv")
data.drop(columns=['Unnamed: 0'],inplace=True)

print(data.head())
location_list= sorted(data["location"].unique())
bath = sorted(data["bath"].unique())
bhk = sorted(data["bhk"].unique())

st.header('Bangalore House Price Predictor',width="content")

Location = st.selectbox("Select the location in Bangalore",location_list,width="stretch")
Area = st.number_input("Enter the Area in Sqrt",width="stretch")
Bhk = st.selectbox("Select the No. of room",bhk,width="stretch")
Bath = st.selectbox("Select the No. of Bathroom",bath,width="stretch")

if st.button("Predict"):
     res =model.predict(pd.DataFrame([[Location,Area,Bhk,Bath]],columns=['location','total_sqft','bhk','bath']))
     # rs = st.header(math.floor(res[0]),width="stretch")
     leng = len(str(math.floor(res[0])))
     if leng==2:
      st.header(f"₹ {math.floor(res[0])} lakh",width="stretch",divider="red")
     if leng==3 :
       st.header(f"₹ {math.floor(res[0])/100} crore", width="stretch", divider="red")
