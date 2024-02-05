import streamlit as st
import os
import openai
import PIL
from PIL import Image
import pandas as pd 


def idea_generator(topic,doc_type):
  response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  max_tokens = 75,
  messages=[
      {"role": "user", "content": "Give me some ideas for what i can do for"+topic+"on"+doc_type}
            
   ]
)
  return response['choices'][0]['message']['content'] + " ... TO GET THE COMPLETE DASHBOARD FEATURES SCHEDULE A CONSULTATION WITH POCKETNINJAS!"


def content_generator(topic,doc_type):
  response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  max_tokens = 250,
  messages=[
      {"role": "user", "content": "Generate a "+doc_type+"on "+topic}
            
   ]
)
  return response['choices'][0]['message']['content'] + " ... TO GET THE COMPLETE DASHBOARD FEATURES SCHEDULE A CONSULTATION WITH POCKETNINJAS!"


def additional_content_generator(doc_type,content):
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      max_tokens = 75,
      messages=[
          {"role": "user", "content": "Generate a "+doc_type+"for this content: "+content}
                
       ]
    )
    return response['choices'][0]['message']['content'] + " ... TO GET THE COMPLETE DASHBOARD FEATURES SCHEDULE A CONSULTATION WITH POCKETNINJAS!"


def image_generator(Img_prompt):
    response = openai.Image.create(
    prompt=Img_prompt,
      n=1,
      size="1024x1024"
    )
    image_url = response['data'][0]['url']

    return image_url

###################################################################

st.set_page_config(page_title="Generate", page_icon="⚙️",layout='wide')


#openai.api_key = "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"

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
            
st.markdown(hide_streamlit_style, unsafe_allow_html=True)    


# Now we create the frontend menu code and all


logo = Image.open('Company_logo.png')

st.sidebar.image(logo)

st.sidebar.markdown("# Welcome to the Content Generation Dashboard.")


# Frontend Code

#image = Image.open('frontend.png')

#st.image(image,caption='Powered by ChatGPT')

st.title("Welcome to the BoxTech Content Generation Dashboard")

st.info('This section can be used to generate ideas for your content along with generating personalized marketing content with a click of a button. ', icon="ℹ️")


### SECTION 1 ###

with st.form("my_form_1"):
   st.subheader("1. I want to generate ideas for content about")
   topic = st.text_input("")
   st.subheader("with the document type being: ")
   doc_type = st.selectbox(
    ' ',
    ('Blog', 'Social Media Post', 'Email newsletter', 'Website Content'))
   
   # Every form must have a submit button.
   submitted = st.form_submit_button("Generate ideas!",use_container_width=True)
   
   if submitted:
       
       st.write("Generating ideas using the GPT AI...")
       
       st.write(idea_generator(topic, doc_type))
       
       
 ### SECTION 2 ###      
       
with st.form("my_form_2"):
   st.subheader("2. Generate a")
   doc_type = st.selectbox(
    ' ',
    ('Blog', 'Social Media Post', 'Email newsletter', 'Website Content'))
   st.subheader("with the topic being: ")
   topic = st.text_input("")
   
   # Every form must have a submit button.
   submitted = st.form_submit_button("Generate content!",use_container_width=True)
   
   if submitted:
       
       st.write("Generating content using the GPT AI...")
       
       st.write(content_generator(topic, doc_type))
       
       
              
### SECTION 3 AND 4 ###


st.info('Now we can proceed to create additional content items.', icon="ℹ️")


col1, col2, = st.columns(2)

with col1:
    with st.form("my_form_3"):
    
       st.subheader("3. Generate")
       additional_content_type = st.selectbox(
        ' ',
        ('Meta Descriptions', 'Titles', 'Subject Lines', 'Tags'))
       
       st.subheader("for the content: ")
       content = st.text_area("",height=400)
       
       
       submitted = st.form_submit_button("Generate additional content!",use_container_width=True)
       
       if submitted:
           
           st.write("Generating additional content using the GPT AI...")
           
           st.write(additional_content_generator(additional_content_type, content))
       
      

with col2:
    
    with st.form("my_form_4"):
        
       st.subheader("4. Generate images for the topic")
       topic = st.text_input("")
       
       
       submitted = st.form_submit_button("Generate image content!",use_container_width=True)
       
       if submitted:
           
           st.write("Generating image using the GPT AI...")
           
           img_url = image_generator(topic)
           
           st.write(img_url)
           
           
       


        
    