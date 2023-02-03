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

st.title('Skin mole type identification')
st.write('Problem definition .... ')

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

# model = keras.models.load_model('mnist_model', compile=False)

# if file_uploader:
#     image = Image.open(file_uploader)
#     st.image(image, caption='Selected Image')

# if st.button('Predict'):
#     #add warning for image not selected
#     image = np.asarray(image)

#     pred = model.predict(image.reshape(1,224,224,3))

    
#     my_bar = st.progress(0)
#     with st.spinner('Predicting'):
#         time.sleep(2)
    
    # if pred == 1:
        # st.write("The mole looks concerning and needs medical attention")
    # else:
        # st.write("The mole looks concerning and needs medical attention")    