
import streamlit as st
import os
import openai
import PIL
from PIL import Image
import pandas as pd 


st.set_page_config(page_title="Implement", page_icon="🎯",layout="wide")

def implementor(message):
  response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
      {"role": "user", "content": message}
            
   ]
)
  return response['choices'][0]['message']['content']


########################


openai.api_key = "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"

hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)    


logo = Image.open('Company_logo.png')

st.sidebar.image(logo)

st.sidebar.markdown("# Welcome to the Implementation Dashboard.")

st.title("Welcome to the BoxTech Implementation Dashboard")

st.info('This section can be used to take action steps using your data', icon="ℹ️")

command = st.text_area(" Enter your desired command here!",placeholder="Send the following information in a email newsletter using mailchimp")

if command:
    
    response = implementor(command)
    st.write(response)
    
    