import streamlit as st
from queue import Empty
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
import sys
sys.tracebacklimit = 0
scaler = StandardScaler()

st.set_page_config(
    layout="wide",
    page_title="Prediction",
    page_icon="❓",
)
with open('style1.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

model = pickle.load(open('model_final.sav','rb'))
 
st.title('Patient\'s data')
image = Image.open('pred_img.png')
new_image = image.resize((900, 250))
st.image(new_image)

#FUNCTION

new_data = {}

def user_report():
    left, right = st.columns(2)   
    with left:     
    # add element on the left side 
        age = float(st.slider('Your age',min_value=1,max_value=100,step=1))
        gender = st.radio('Your Gender', ['Male', 'Female'],horizontal=True) 
        hypertension = st.radio('Do you have Hypertension?', ['Yes', 'No'],horizontal=True)
        heart_disease = st.radio('Do you have a Heart Disease?', ['Yes', 'No'],horizontal=True)
        ever_married = st.radio('Were you Ever Married or are married?', ['Yes', 'No'],horizontal=True)  
    with right:     
  # add element on the right side
        work_type = st.radio('Select your Work Type' , ['Private','Self-employed','Children (below 18 yrs age)','Never-worked'],horizontal=True)
        Residence_type = st.radio('Select the type of your Residence',['Urban','Rural'],horizontal=True)
        smoking_status = st.radio('Smoking history',['never smoked','formerly smoked','smokes'],horizontal=True)
        avg_glucose_level =float (st.number_input('Your Average glucose level',min_value=0.0))
        bmi = float(st.number_input('Your current BMI',min_value=0.0))

    if(age >=18 and work_type=='Children (below 18 yrs age)'):
        st.error('Please check your age and worktype, either of them is incorrect')
        raise Exception('Please check your age and worktype, either of them is incorrect')
    
    new_data = {
        'gender':gender,
        'age':age,
        'hypertension':hypertension,
        'heart_disease':heart_disease,
        'ever_married':ever_married,
        'work_type':work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level':avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    if(hypertension=='Yes'):
        hypertension=1
    else:
        hypertension=0

    if(heart_disease=='Yes'):
        heart_disease=1
    else:
        heart_disease=0

    if(gender =='Male'):
        gender_Male = 1
    else:
        gender_Male = 0

    if(ever_married=='Yes'):
        ever_married_yes=1
    else:
        ever_married_yes=0

    if(Residence_type=='Urban'):
        residence=1
    else:
        residence=0

    if(smoking_status=='formerly smoked'):
        smoking_status_formerlysmoked =1
        smoking_status_neversmoked=0
        smoking_status_smokes=0
    elif(smoking_status=='never smoked'):
        smoking_status_formerlysmoked =0
        smoking_status_neversmoked=1
        smoking_status_smokes=0
    else:
        smoking_status_formerlysmoked =0
        smoking_status_neversmoked=0
        smoking_status_smokes=1

    if(work_type=='Private'):
        work_type_Private=1
        work_type_never_worked=0
        work_type_Self_employed =0
        work_type_children=0

    elif(work_type=='Self-employed'):
        work_type_Private=0
        work_type_never_worked=0
        work_type_Self_employed =1
        work_type_children=0
    elif(work_type=='Children (below 18 yrs age'):
        work_type_Private=0
        work_type_never_worked=0
        work_type_Self_employed =0
        work_type_children=1
    else:
        work_type_Private=0
        work_type_never_worked=1
        work_type_Self_employed =0
        work_type_children=0
    
    age=float(age)
    avg_glucose_level = float(avg_glucose_level)
    bmi = float(bmi)

    base_df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    base_con_cols = ['age','avg_glucose_level','bmi']


    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    con_cols = np.array([age,avg_glucose_level,bmi])
    con_cols = con_cols.reshape(1,-1)
  
    scaler.fit(base_df[base_con_cols])
    
    con_cols=scaler.transform(con_cols)
   
    age = con_cols[0][0]
    avg_glucose_level = con_cols[0][1]
    bmi = con_cols[0][2]



    user_report_data = {
        'age':age,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender_Male' : gender_Male,
        'ever_married_Yes': ever_married_yes,
        'work_type_Never_worked':work_type_never_worked,
        'work_type_Private':work_type_Private,
        'work_type_Self-employed' :work_type_Self_employed,
        'work_type_children':work_type_children,
        'Residence_type_Urban': residence,
        'smoking_status_formerly smoked': smoking_status_formerlysmoked,
        'smoking_status_never smoked': smoking_status_neversmoked,
        'smoking_status_smokes': smoking_status_smokes
    }

    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data


user_data = user_report()

stroke = model.predict(user_data)
st.header('Patient\'s Data after preprocessing')
st.write(user_data) 

p_df = pd.read_csv('pre_d.csv')


st.header('Do you have a chances of stroke in future?')
if st.button('PREDICT'):
        if stroke==1:
            st.markdown(':red[You have fair chances of having a stroke in future. Please take necessary precautions and consult your doctor. Some of the precautions are: ]')
            user_data['stroke']=1
            #st.markdown('<p class="big-font"> <ol > <li> Choose healthy foods and drinks </li> <li> Keep a healthy weight.</li> <li> Get regular physical activity. </li> <li> Quit smoking and alcohol.</li> </ol></p>', unsafe_allow_html=True)

            st.subheader(" 1. Choose healthy foods and drinks")
            st.subheader(" 2. Keep your weight under control." )
            st.subheader(" 3. Do regular physical activity.")
            st.subheader(" 4. Avoid smoking and drinking alcohol. ")
            p_df = p_df.append(user_data, ignore_index = True)
            p_df.to_csv('pre_d.csv', index = False) 
        else:   
            user_data['stroke']=0
            st.markdown(':green[The model predicts that you do not have probable chances of having stroke in future, so no need to worry :) But keeping a healthy lifestyle is always beneficial.]')
            p_df = p_df.append(user_data, ignore_index = True)
            p_df.to_csv('pre_d.csv', index = False)
# st.subheader('thanks')

