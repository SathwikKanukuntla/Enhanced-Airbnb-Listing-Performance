   # -*- coding: utf-8 -*-


import streamlit as st 
import numpy as np
import os
import openai
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import pandas as pd 
import joblib
from keras.models import load_model

#testing

import plotly.express as px




#st.set_page_config(page_title="BoxTech", page_icon="📈" , layout='wide' )


def data_analyzer(data,message):
  response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
      {"role": "user", "content": message + "for the data" + f"{data}"}
            
   ]
)
  return response['choices'][0]['message']['content']




# Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

# Streamlit set_page_config method has a 'initial_sidebar_state' argument that controls sidebar state.
st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state,page_title="BoxTech", page_icon="📈" , layout='wide' )

# Show title and description of the app.
st.sidebar.markdown('Choose the tool you need!')

# Toggle sidebar state between 'expanded' and 'collapsed'.
if st.button('Click to toggle sidebar state'):
    st.session_state.sidebar_state = 'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
    # Force an app rerun after switching the sidebar state.
    st.experimental_rerun()




########################


openai.api_key = "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"

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
            
st.markdown(hide_streamlit_style, unsafe_allow_html=True)    


# Now we create the frontend menu code and all


logo = Image.open('Company_logo.png')

st.sidebar.image(logo)

st.sidebar.markdown("# Welcome to your Analytics Dashboard.")


# Frontend Code

#image = Image.open('frontend.png')

#st.image(image,caption='Powered by ChatGPT')

st.title("Welcome to the BoxTech Analytics Dashboard")

st.info('BoxTech is an example startup that uses this dashboard to analyze its marketing information and make decisions using ChatGPT4, the latest AI technology.', icon="ℹ️")

st.header("Business Summary")


st.write("---------------------------------------------")

#########################################################

################### Creating the knobs to control the charts ################


st.info('Take a call up with us and get your custom dashboards built on your company data. We have integrations with AWS, Google Cloud, Excel sheets and many more. Over here we show BoxTechs data visualized on a simple interface', icon="ℹ️")


X_axis,Y_axis,Aggregation_level = st.columns(3)



X_axis = X_axis.selectbox('Choose your X axis value',('Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
       'Teenhome', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Complain', 'Response'))

Aggregation_level = Aggregation_level.selectbox('Choose the level of aggregation for the Y axis',('All Points', 'Average', 'Sum','Count','Max','Min'))


Y_axis = Y_axis.selectbox('Choose your Y axis value',('Income', 'Kidhome',
       'Teenhome', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Complain', 'Response'))




###########Analytics Dashboard###########################



data= pd.read_csv("marketing_campaign.csv",delimiter=';')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

if X_axis not in ['Education', 'Marital_Status']:
    
    

    
    if Aggregation_level == 'All points':
        fig = px.scatter(data_frame=data,x=X_axis,y=Y_axis,width=800, height=700,trendline="lowess")
        fig.update_traces(marker={'size': 15})
    
    elif Aggregation_level == 'Average':
        fig = px.scatter(data_frame=data.groupby([X_axis]).mean().reset_index(),x=X_axis,y=Y_axis,width=800, height=700,trendline="lowess")
        fig.update_traces(marker={'size': 15})
        
    elif Aggregation_level == 'Sum':
        fig = px.scatter(data_frame=data.groupby([X_axis]).sum().reset_index(),x=X_axis,y=Y_axis,width=800, height=700,trendline="lowess")
        fig.update_traces(marker={'size': 15})
    
    elif Aggregation_level == 'Count':
        fig = px.scatter(data_frame=data.groupby([X_axis]).count().reset_index(),x=X_axis,y=Y_axis,width=800, height=700,trendline="lowess")
        fig.update_traces(marker={'size': 15})
        
    
    elif Aggregation_level == 'Max':
        fig = px.scatter(data_frame=data.groupby([X_axis]).max().reset_index(),x=X_axis,y=Y_axis,width=800, height=700,trendline="lowess")
        fig.update_traces(marker={'size': 15})
    
    
    elif Aggregation_level == 'Min':
        fig = px.scatter(data_frame=data.groupby([X_axis]).min().reset_index(),x=X_axis,y=Y_axis,width=800, height=700,trendline="lowess")
        fig.update_traces(marker={'size': 15})
        
    
    else:
        fig = px.scatter(data_frame=data,x=X_axis,y=Y_axis,width=800, height=700,trendline="lowess")
        fig.update_traces(marker={'size': 15})



else:
    
    if Aggregation_level == 'All points':
        fig = px.bar(data_frame=data,x=X_axis,y=Y_axis,width=800, height=700,trendline="lowess")
    
    elif Aggregation_level == 'Average':
        fig = px.bar(data_frame=data.groupby([X_axis]).mean().reset_index(),x=X_axis,y=Y_axis,width=800, height=700)
        
    
    elif Aggregation_level == 'Sum':
        fig = px.bar(data_frame=data.groupby([X_axis]).sum().reset_index(),x=X_axis,y=Y_axis,width=800, height=700)
    
    elif Aggregation_level == 'Count':
        fig = px.bar(data_frame=data.groupby([X_axis]).count().reset_index(),x=X_axis,y=Y_axis,width=800, height=700)
        
    
    elif Aggregation_level == 'Max':
        fig = px.bar(data_frame=data.groupby([X_axis]).max().reset_index(),x=X_axis,y=Y_axis,width=800, height=700)
    
    
    elif Aggregation_level == 'Min':
        fig = px.bar(data_frame=data.groupby([X_axis]).min().reset_index(),x=X_axis,y=Y_axis,width=800, height=700)
        
    
    else:
        fig = px.bar(data_frame=data,x=X_axis,y=Y_axis,width=800, height=700)

    

fig.update_layout(
yaxis = dict(
tickfont = dict(size=20))
,xaxis = dict(
tickfont = dict(size=20)))

fig.update_yaxes(title_font=dict(size=12))
fig.update_xaxes(title_text=X_axis)
fig.update_yaxes(title_text=Y_axis)

st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme='streamlit')
    
    
   
st.write("---------------------------------------------")



st.header("Ask the AI questions about your data")


st.info('Ex: "What is the average income based on the category of education? Show me a bar plot."', icon="ℹ️")


analysis_prompt = st.text_area(" Enter your question here!")

"""

import os
os.environ['OPENAI_API_KEY'] = "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"

from llama_index import download_loader
from pandasai.llm.openai import OpenAI
import pandas as pd
llm = OpenAI()

PandasAIReader = download_loader("PandasAIReader")

reader = PandasAIReader(llm=llm)
response = reader.run_pandas_ai(
    data, 
    analysis_prompt, 
    is_conversational_answer=False
)
"""

response = data_analyzer(data,analysis_prompt)

st.write(response)


# Create a function that takes in a dataframe w a dataframe argument and an analysis prompt. 
# And returns the value of the output


st.write("---------------------------------------------")



st.header("Use AI predictor to train on recent data and give you predictions")


st.info('This particular business has trained its data to predict the success of a new campaign based on historical campaign data. This can be customized as per your needs.', icon="ℹ️")


# Year_Birth, Income, Kidhome
# Teenhome, Recency, MntWines
# MntFruits, MntMeatProducts, MnsFishProducts
# MntSweetProducts, MntGoldProds, NumDealsPurchases
# NumWebPurchases, NumCatalogPurchases,NumStorePurchases
# NumWebVisitsMonth, AcceptedCmp3, AcceptedCmm4
# AcceptedCmp5, AcceptedCmp1, AcceptedCmp2
# Complain, Response, Education
# Marital_status


yes_no_bool = {"Yes":1,"No":0}

with st.expander("Predict the success of the new campaign"):
    Year_Birth,Income,Kidhome = st.columns(3)

    Year_Birth = Year_Birth.number_input("Birth Year of Customer",step=1,min_value=1800,max_value=2000)
    Income = Income.number_input("Income of individual")
    Kidhome = Kidhome.radio("Do they have small children",["Yes","No"])
    Kidhome = yes_no_bool[Kidhome]
    
    st.write("---------------------------------------------")
    ##########################################################
    
    Teenhome, Recency, MntWines = st.columns(3)
    
    Teenhome = Teenhome.radio("Do they have teenagers",["Yes","No"])
    Teenhome = yes_no_bool[Teenhome]
    Recency = Recency.number_input("Number of days since the last purchase")
    MntWines = MntWines.number_input("Amount of money spent on wines")
    
    st.write("---------------------------------------------")
    ##########################################################
    
    MntFruits, MntMeatProducts, MntFishProducts = st.columns(3)
    
    MntFruits = MntFruits.number_input("Amount of money spent on fruits")
    MntMeatProducts = MntMeatProducts.number_input("Amount of money spent on meat")
    MntFishProducts = MntFishProducts.number_input("Amount of money spent on fish")
    
    st.write("---------------------------------------------")
    ##########################################################
    
    MntSweetProducts,MntGoldProds,NumDealsPurchases = st.columns(3)
    
    MntSweetProducts = MntSweetProducts.number_input("Amount of money spent on sweet products")
    MntGoldProds = MntGoldProds.number_input("Amount of money spent on gold")
    NumDealsPurchases = NumDealsPurchases.number_input("Purchases using discount")
    
    st.write("---------------------------------------------")
    ##########################################################
    
    # We got to scale these values accordingly.
    
    scaler = joblib.load('scaler.save') 
    
    
    
    data_instance=[Year_Birth, Income, Kidhome,Teenhome, Recency, MntWines,
                   MntFruits, MntMeatProducts, MntFishProducts,MntSweetProducts, MntGoldProds, NumDealsPurchases
                   ,4.0,7.0,11.0,2.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0]
    
    
    data_instance=np.array(data_instance).reshape(1,-1)
    
    data_instance_transformed = scaler.transform(data_instance)
    
    model=load_model('my_model.h5')
    
    
   # data=np.array(data_instance_transformed).reshape(1,-1)
    
    prediction = model.predict(data_instance_transformed)
    
    
    prediction_final=np.where(prediction > 0.5, 1, 0)
    
   
    prediction_converter = {0:"Less likely to purchase",1:"More likely to purchase"}
    
   
    prediction_final=prediction_converter[prediction_final[0][0]]
    
    
    st.write("Predict the chances customer purchasing. This will update depending on the settings")
    
    st.success(prediction_final)
   
    

   



















################ Create a predictor #####################













    
