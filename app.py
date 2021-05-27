import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('CIFAR_model.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

html_temp = """
<div style="background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;">Image Classifier model Developed using CNN</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.text("")
st.markdown("***")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data):
    size = (32,32)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    img_reshape = img[np.newaxis,...]
    result = model.predict(img_reshape)
    return result

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image)
    cls = np.argmax(predictions)
    if cls == 0:
        st.success("Prediction : Its an image of Airoplane.")
    elif cls == 1:
        st.success("Prediction : Its an image of Automobile.")
    elif cls == 2:
        st.success("Prediction : Its an image of Bird.")
    elif cls == 3:
        st.success("Prediction : Its an image of Cat.")
    elif cls == 4:
        st.success("Prediction : Its an image of Deer.")
    elif cls == 5:
        st.success("Prediction : Its an image of Dog.")
    elif cls == 6:
        st.success("Prediction : Its an image of Frog.")
    elif cls == 7:
        st.success("Prediction : Its an image of Horse.")
    elif cls == 8:
        st.success("Prediction : Its an image of Ship.")
    elif cls == 9:
        st.success("Prediction : Its an image of Truck.")

    # st.success("This image most likely belongs to {} .".format(classes[np.argmax(predictions)]))
