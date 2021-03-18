#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow import keras
import cnn_model
import Seq_model
import pandas as pd
import pickle
import time
#import Demo Images
#from sequential_model import history_graph

import numpy as np
from PIL import Image, ImageOps

#st.set_page_config(page_title="Clothes Classification App", page_icon="fashion_icon.jpg", layout='centered', initial_sidebar_state='auto')
    

st.markdown("""
    <style>
    .reportview-container {
        background: url("https://source.unsplash.com/1600x900/?sky,cloud")
    }
   .sidebar .sidebar-content {
        background: url("")
    }
    </style>
    """,
    unsafe_allow_html=True
)


# In[3]:
fas_data=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fas_data.load_data()




seq_model=tf.keras.models.load_model("Seq_model")
cnn_model=tf.keras.models.load_model("cnn_model")
class_names=['Tshirt/TOP','Trouser','Pullover','Dress','Coat',
             'Sandel','Shirt','Sneaker','Bag','Ankle boot']


# In[4]:


######### -------------- Sidebarr--------------------->
add_selectbox = st.sidebar.selectbox(
    'select the model for classification',
    ('Sequential','CNN','About data','Pretrained Neural network','Demo Images','Working Demo','Contact us')
)


st.title("Fashion MNIST dataset Classification")


# In[5]:


import numpy as np
def explore_data(train_images,train_label,test_images):
    st.write('Train Images shape:',train_images.shape)
    st.write('Test images shape:',test_images.shape)
    #st.write(train_labels[0:20])
    st.write('Training Classes',len(np.unique(train_labels)))
    st.write('Testing Classes',len(np.unique(test_labels)))

def  CNN_model_summary():
    img=Image.open("cnn_summary.PNG")
    st.image(img)
def  Seq_model_Summary():
    img=Image.open("Seq_summary.PNG")
    st.image(img)
    
def seq_history_graph():
    infile=open('seq_trainHistory',"rb")
    history = pickle.load(infile)
    plt.figure(figsize=(7,7))
    train_acc=history['accuracy']
    val_acc=history['val_accuracy']
    train_loss=history['loss']
    val_loss=history['val_loss']
    plt.subplot(2,1,1)
    plt.plot(train_acc,label='Training accuracy')
    plt.plot(val_acc,label='Validation accuracy')
    plt.legend()
    plt.title('acc')
    plt.subplot(2,1,2)
    plt.plot(train_loss,label='Training loss')
    plt.plot(val_loss,label='Validation loss')
    plt.legend()
    plt.title('loss')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
                
    
def cnn_history_graph():
    infile=open('cnntrainHistory',"rb")
    history = pickle.load(infile)
    plt.figure(figsize=(7,7))
    train_acc=history['accuracy']
    val_acc=history['val_accuracy']
    train_loss=history['loss']
    val_loss=history['val_loss']
    plt.subplot(2,1,1)
    plt.plot(train_acc,label='Training accuracy')
    plt.plot(val_acc,label='Validation accuracy')
    plt.legend()
    plt.title('acc')
    plt.subplot(2,1,2)
    plt.plot(train_loss,label='Training loss')
    plt.plot(val_loss,label='Validation loss')
    plt.legend()
    plt.title('loss')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
    
def cnn_archi():
    img=Image.open('cnn_model_architecture.png')
    st.image(img)
def seq_archi():
    img=Image.open('seq_model_architecture.png')
    st.image(img)


def about_data(cnn_model,Seq_model):
    
    if st.button("Explore Data"):
        explore_data(train_images,train_labels,test_images)
    
    if st.button('CNN ModelSumarry'):
        
        CNN_model_summary()
    if st.button('CNN Model Architecture'):
        cnn_archi()
    if st.button('Seq ModelSumarry'):
        Seq_model_Summary()
        
    if st.button('Sequntial model Architecture'):
        seq_archi()
 
 
    
    if st.button('sequential model graph'):
        seq_history_graph()
    if st.button('CNN model graph'):
        cnn_history_graph()

if add_selectbox=='Demo Images':
  st.write("Please Upload the Following type of images of cloths for classification")
  image=Image.open("Demo Images/shirt.jpeg")
  image=image.resize((180,180))
  st.image(image)
  image1=Image.open("Demo Images/bag.jpg")
  image1=image1.resize((180,180))
  st.image(image1)
  image2=Image.open("Demo Images/sneaker.jpg")
  image2=image2.resize((180,180))
  st.image(image2)
  image3=Image.open("Demo Images/t-shirt.jfif")
  image3=image3.resize((180,180))
  st.image(image3)
  image4=Image.open("Demo Images/blazer.jpg")
  image4=image4.resize((180,180))
  st.image(image4)
  image5=Image.open("Demo Images/pant.jpg")
  image5=image5.resize((180,180))
  st.image(image5)
                   
  
    
    
if add_selectbox=='About data':
    about_data(cnn_model,Seq_model)

if add_selectbox=='Pretrained Neural network':
    st.info("working on it, updated soon!")
if add_selectbox=='Working Demo':
    video_file = open('fashion-working-demo.webm', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    
if(add_selectbox=='Contact us'):
    image=Image.open('sandeep yadav.jpg')
    image=image.resize((400,400))
    st.image(image)
    st.write('Sandeep Yadav')
    st.write('contact:sandeep18498@gmail.com')



# file uploder

if(add_selectbox=='CNN' or add_selectbox=='Sequential'):
    file_uploader=st.file_uploader('Upload cloth Image for Classification:')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file_uploader is not None:
        image=Image.open(file_uploader)
        text_io = io.TextIOWrapper(file_uploader)
        image=image.resize((180,180))
        st.image(image,'Uploaded image:')
        
           
        def classify_image(image,model):
            st.write("classifying......")
            img = ImageOps.grayscale(image)
        
            img=img.resize((28,28))
            if(add_selectbox=='Sequential'):
                img=np.expand_dims(img,0)
            else:
                img=np.expand_dims(img,0)
                img=np.expand_dims(img,3)
            img=(img/255.0)
                  
            img=1-img
        
        
            pred=model.predict(img)
        
            st.write("The Predicted image is:",class_names[np.argmax(pred)])
            st.write('Prediction probability :{:.2f}%'.format(np.max(pred)*100))
        st.write('Click for classify the image')
        if st.button('Classify Image'):
            if(add_selectbox=='Sequential'):
                st.write("You are choosen Image classification with Sequential Model")
                classify_image(image,seq_model)
                st.success('This Image successufully classified!')
                with st.spinner('Wait for it...'):
                    time.sleep(2)
                    st.success('Done!')
                    st.balloons()
                st.balloons()
            if(add_selectbox=='CNN'):
                st.write("You are choosen Image classification with CNN Model")
                classify_image(image,cnn_model)
                st.success('This Image successufully classified!')
                with st.spinner('Wait for it...'):
                    time.sleep(2)
                    st.success('Done!')
                    st.balloons()
    else:
        st.write("Please select image:")

        

# In[ ]:




