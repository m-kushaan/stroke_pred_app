import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config( 
    page_title="Stroke Prediction WebApp",
    page_icon="♥",
    layout="wide",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


left, right = st.columns(2)
with left:
    st.title('STROKE PREDICTION WEBAPP')
    with st.container():
        st.write('Stroke occurs when the blood flow to various areas of the brain is disrupted or diminished, resulting in the cells in those areas of the brain not receiving the nutrients and oxygen they require and are dying. A stroke is a medical emergency that requires urgent medical attention. Early detection and appropriate and timely management are required to prevent further damage to the affected area of the brain and other complications in other parts of the body. ')
        st.write('To predict whether you have chances of having stroke in near future, this webapp will help you predict the chances with 97% accuracy. Remember, prevention is better than cure.')
with right:
    image = Image.open('main_img.jpg')
    new_image = image.resize((470, 470))
    st.image(new_image)
    # st.image(image,'')

st.header('Training dataset used: ')
base_df = pd.read_csv("healthcare-dataset-stroke-data.csv")
st.write(base_df)



