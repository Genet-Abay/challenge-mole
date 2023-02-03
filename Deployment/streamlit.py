import streamlit as st
import pandas as pd
import altair as alt
import keras
from PIL import Image
import numpy as np
from keras.utils.vis_utils import plot_model
import seaborn as sns
import time
from matplotlib import pyplot as plt
import cv2

st.title('Skin mole type identification')

st.write('This application is an AI model that classifies a given image as either cancerous or not. The model accuray and loss over epochs are shown bellow')

df = pd.read_csv(r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_metadata.csv")
df['concern'] = df['dx'].apply(lambda x : 0 if ((x == 'nv') | (x == 'bkl') | (x == 'df') | (x == 'vasc')) else 1)
df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 100], labels=["0-20", "20-40", "40-60", "60-80", "80-100"])


fig = plt.figure(figsize=(6,3))
sns.countplot(x=df["age_bin"], hue=df['concern'])
plt.title('Skin mole concern on ages')
plt.xlabel('age')
st.pyplot(plt)

st.subheader('Please upload imaget')
file_uploader = st.file_uploader("Select an image for Classification", type="jpg")
print(file_uploader)

model = keras.models.load_model('Deployment/model_keras.h5', compile=False)
 

if file_uploader:
    image = Image.open(file_uploader).resize((128,128))
    st.image(image, caption='Selected Image and predict')
    plt.imshow(image)

    # img_test_ex = cv2.resize(image, (128,128))
    img_test_ex = np.asarray(image)
    img_test_ex = np.array(img_test_ex/255.0)
    img_test_ex = np.expand_dims(img_test_ex, axis=0)
    result = model.predict(img_test_ex)

    
    my_bar = st.progress(0)
    with st.spinner('Predicting'):
        time.sleep(2)
    
    if  ((result == 'nv') | (result == 'bkl') | (result == 'df') | (result == 'vasc')):
        print("the images looks of no concern, not cancerous")
    else:
        print("the image is classified as cancerous mole and needs medical attention")