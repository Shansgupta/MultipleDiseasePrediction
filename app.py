import pickle
import streamlit as st
from streamlit_option_menu import option_menu  

diabetes_model = pickle.load(open("C:/MultipleDiseasePrediction/artifacts/diabetes2.pkl","rb"))
heart_model = pickle.load(open("C:/MultipleDiseasePrediction/artifacts/heart_disease.pkl","rb"))

with st.sidebar :
    selected = option_menu('Mutiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction'],
                            icons = ['activity','heart'],
                           default_index=0)

if (selected == 'Diabetes Prediction'):

    st.title('Diabetes Prediction using ML')
    gender = st.text_input("Gender")
    age = st.text_input("Age")
    hypertension = st.text_input(" Hypertension value")
    heart_disease = st.text_input("heart disease value")
    smoking_history = st.text_input("Smoking  : never , unknown ,current ,past")
    bmi = st.text_input("Bmi value")
    HbA1c_level = st.text_input("HbA1c level")
    blood_glucose_level = st.text_input("Blood Glucose level")
    if  age.strip().isdigit():
      age = int(age)  # ✅ Convert to integer

    # ✅ Use it for age_group classification
    if age < 13:
        age_group = "Child"
    elif 13 <= age < 18:
        age_group = "Teenager"
    elif 18 <= age < 60:
        age_group = "Adult"
    else:
        age_group = "Senior"

    st.success(f"Age group automatically detected: **{age_group}**")

else:
    st.warning("Please enter a valid number for age.")
    age_group = None
    glucose_hba1c = st.text_input("value")
    bmi_bin = st.text_input("value1")
    age_bmi_interaction = st.text_input("Enter value")
    bmi_glucose_interaction = st.text_input("Enter a")





if (selected == 'Heart Disease Prediction'):
     
     st.title('Heart Disease Prediction using ML')
         