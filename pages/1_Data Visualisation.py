from pickletools import int4
import statistics
import streamlit as st
from audioop import avg
from queue import Empty
from selectors import EVENT_READ
import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np 
import seaborn as sns
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    layout="wide",
    page_title="Visualization",
    page_icon="📈",
)

st.title("VISUALISATION OF THE TRAINING DATASET")

st.header("Gender Ratio in the data samples and their relation with BMI")

base_df = pd.read_csv("healthcare-dataset-stroke-data.csv")

fig = plt.figure(figsize=(10, 4))
sns.boxplot(x=base_df['gender'], y= base_df['bmi'])
st.pyplot(fig)

st.header("Scatterplot of BMI vs age")


fig = plt.figure(figsize=(10, 4))
sns.scatterplot(data = base_df, x='bmi', y='age')
st.pyplot(fig)

age_grp=[]

for i in base_df['age']:
    if i<2:
         age_grp.append('Toddler')
    elif i>2 and i<=19:
        age_grp.append('Teen')
    elif i>19 and i<60:
        age_grp.append('Adult')
    else:
        age_grp.append('Senior')
base_df['age_group']=age_grp
base_df.head()

st.header("Box plot of age_group vs BMI")

fig = plt.figure(figsize=(10, 4))
sns.boxplot(x=base_df["age_group"], y=base_df["bmi"])
st.pyplot(fig)

st.header("Heatmap")
sns.heatmap(base_df.corr(),annot=True,linewidth=0.5,fmt='0.2f')
st.pyplot(fig)




