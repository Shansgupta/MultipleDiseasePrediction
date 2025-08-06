import streamlit as st
import pandas as pd
import pickle
import os
from streamlit_option_menu import option_menu

# --- Page Configuration ---
# This should be the first Streamlit command in your script.
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# --- Function to Load Files ---
# This function loads your .pkl files and handles errors if a file is missing.


# --- Load Your Pre-trained Models and Preprocessors ---
heart_preprocessor = pickle.load(open('C:/MultipleDiseasePrediction/artifacts/heart_preprocessor.pkl','rb'))
diabetes_model = pickle.load(open('C:/MultipleDiseasePrediction/artifacts/diabetes_diseases.pkl','rb'))
diabetes_preprocessor = pickle.load(open('C:/MultipleDiseasePrediction/artifacts/diabetes_preprocessor.pkl','rb'))
heart_model = pickle.load(open('C:/MultipleDiseasePrediction/artifacts/heart_diseases.pkl','rb'))


# --- Sidebar for Navigation ---
with st.sidebar:
    st.title("Navigation")
    selected  = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction"],
        default_index=0
    )


# --- Diabetes Prediction Page ---
if selected == "Diabetes Prediction":
    st.title("🩺 Diabetes Prediction")

    # We only show the UI if the files were loaded correctly
    if diabetes_preprocessor and diabetes_model:
        st.markdown("Enter the patient's details below.")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female', 'Other'], key='d_gender')
            if gender == "Male":
              gender = 0
            elif gender == "Female":
             gender = 1
            else:
              st.error("Please enter gender as 'M' or 'F'")
            age = st.number_input('Age', min_value=0, max_value=120, value=54, step=1, key='d_age')
            hypertension = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='d_hyper')
            heart_disease = st.selectbox('Heart Disease History', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='d_heart')

        with col2:
            smoking_history = st.selectbox('Smoking History', ['never', 'unknown', 'current', 'past',], key='d_smoke')
            bmi = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=100.0, value=27.3, step=0.1, key='d_bmi')
            HbA1c_level = st.number_input('HbA1c Level', min_value=3.0, max_value=15.0, value=6.6, step=0.1, key='d_hba1c')
            blood_glucose_level = st.number_input('Blood Glucose Level (mg/dL)', min_value=50, max_value=400, value=140, key='d_glucose')

        if st.button('**Predict Diabetes**', type="primary"):
            # Create a DataFrame from the user's input
            # The column names MUST match the names used when training the preprocessor
            input_data = pd.DataFrame({
                'gender': [gender], 'age': [age], 'hypertension': [hypertension],
                'heart_disease': [heart_disease], 'smoking_history': [smoking_history],
                'bmi': [bmi], 'HbA1c_level': [HbA1c_level], 'blood_glucose_level': [blood_glucose_level]
            })

            def classify_bmi(bmi):
              if bmi < 18.5:
               return "Underweight"
              elif 18.5 <= bmi < 25:
               return "Normal"
              elif 25 <= bmi < 30:
               return "Overweight"
              else:
               return "Obese"

            def get_age_group(age):
              if age <= 18:
               return "child"
              elif age <= 35:
               return "young"
              elif age <= 60:
               return "adult"
              else:
               return "senior"

            input_data["bmi_bin"] = input_data["bmi"].apply(classify_bmi)
            input_data["age_group"] = input_data["age"].apply(get_age_group)
            input_data["glucose_hba1c"] = input_data["blood_glucose_level"] * input_data["HbA1c_level"]
            input_data["age_bmi_interaction"] = input_data["age"] * input_data["bmi"]
            input_data["bmi_glucose_interaction"] = input_data["bmi"] * input_data["blood_glucose_level"]


            # Transform the data using the preprocessor
            transformed_data =  diabetes_preprocessor.transform(input_data)
            
            # Make a prediction
            prediction = diabetes_model.predict(transformed_data)
            prediction_proba = diabetes_model.predict_proba(transformed_data)

            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error(f"**Result: The person is LIKELY to have Diabetes.** (Confidence: {prediction_proba[0][1]*100:.2f}%)")
            else:
                st.success(f"**Result: The person is LIKELY Non-Diabetic.** (Confidence: {prediction_proba[0][0]*100:.2f}%)")

# --- Heart Disease Prediction Page ---
if selected == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction")

    # We only show the UI if the files were loaded correctly
    if heart_preprocessor and heart_model:
        st.markdown("Enter the patient's details below.")
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input('Age', min_value=1, max_value=120, value=54, key='h_age')
            Sex = st.text_input("Gender (M/F)").strip().upper()

           # Convert to numeric for model
            if Sex == "M":
              Sex = 0
            elif Sex == "F":
             Sex = 1
            else:
              st.error("Please enter gender as 'M' or 'F'")
            ChestPainType = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'], key='h_cpt')
        with col2:
            RestingBP = st.number_input('Resting Blood Pressure', min_value=0, max_value=250, value=130, key='h_rbp')
            Cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200, key='h_chol')
            FastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'False' if x == 0 else 'True', key='h_fbs')
        with col3:
            RestingECG = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'], key='h_recg')
            MaxHR = st.number_input('Max Heart Rate Achieved', min_value=50, max_value=250, value=150, key='h_mhr')
            ExerciseAngina = st.selectbox('Exercise Induced Angina', ['N', 'Y'], key='h_exang')
            if ExerciseAngina == "N":
             ExerciseAngina = 0
            elif ExerciseAngina == "Y":
             ExerciseAngina = 1
            else:
              st.error("Please enter ExerciseAngina as 'N' or 'Y'")
               
            Oldpeak = st.number_input('Oldpeak', value=1.0, step=0.1, key='h_oldpeak')
            ST_Slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'], key='h_stslope')

        if st.button('**Predict Heart Disease**', type="primary"):
            # Create a DataFrame from the user's input
            input_data_hd = pd.DataFrame({
                'Age': [Age], 'Sex': [Sex], 'ChestPainType': [ChestPainType], 'RestingBP': [RestingBP],
                'Cholesterol': [Cholesterol], 'FastingBS': [FastingBS], 'RestingECG': [RestingECG],
                'MaxHR': [MaxHR], 'ExerciseAngina': [ExerciseAngina], 'Oldpeak': [Oldpeak], 'ST_Slope': [ST_Slope]
            })
            input_data_hd['Oldpeak_outlier'] = (input_data_hd['Oldpeak'] > 4).astype(int)
            input_data_hd['cholesterol_outlier'] = (input_data_hd['Cholesterol'] > 300).astype(int)
            # Transform the data and make a prediction
            transformed_data_hd = heart_preprocessor.transform(input_data_hd)
            prediction_hd = heart_model.predict(transformed_data_hd)
            prediction_proba_hd = heart_model.predict_proba(transformed_data_hd)

            st.subheader("Prediction Result")
            if prediction_hd[0] == 1:
                st.error(f"**Result: The person is LIKELY to have Heart Disease.** (Confidence: {prediction_proba_hd[0][1]*100:.2f}%)")
            else:
                st.success(f"**Result: The person is LIKELY Healthy.** (Confidence: {prediction_proba_hd[0][0]*100:.2f}%)")
