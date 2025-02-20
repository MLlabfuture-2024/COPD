import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import joblib


def main():
    st.title("Disease Prediction Calculator")

    # Define categorical and numerical features
    cat_features = ['Gender', 'Race', 'Education', 'Smoke', 'Alcohol']
    num_features = ['Age', 'BMI', 'PIR', 'MPAH', 'PFDE', 'PFHxS', 'PFNA', 'PFOA', 'PFOS', 'PFUA']

    # Load model and scaler
    try:
        model = joblib.load('catboost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

    # Main page - Data input
    st.header("Patient Information")

    # Categorical variables input
    st.subheader("Basic Information")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        race = st.selectbox("Race", [1, 2, 3, 4, 5],
                            format_func=lambda x: ["Mexican American", "Other Hispanic",
                                                   "Non-Hispanic White", "Non-Hispanic Black",
                                                   "Other Race"][x - 1])
        education = st.selectbox("Education", [0, 1],
                                 format_func=lambda x: "High school or below" if x == 0 else "Above high school")

    with col2:
        smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        alcohol = st.selectbox("Alcohol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Numerical variables input
    st.subheader("Physical Indicators")
    col3, col4, col5 = st.columns(3)

    with col3:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
        pir = st.number_input("PIR", min_value=0.0, max_value=10.0, value=2.0)

    with col4:
        mpah = st.number_input("MPAH (ng/mL)", min_value=0.0, max_value=100.0, value=0.5)
        pfde = st.number_input("PFDE (ng/mL)", min_value=0.0, max_value=100.0, value=0.5)
        pfhxs = st.number_input("PFHxS (ng/mL)", min_value=0.0, max_value=100.0, value=1.0)
        pfna = st.number_input("PFNA (ng/mL)", min_value=0.0, max_value=100.0, value=0.5)

    with col5:
        pfoa = st.number_input("PFOA (ng/mL)", min_value=0.0, max_value=100.0, value=2.0)
        pfos = st.number_input("PFOS (ng/mL)", min_value=0.0, max_value=100.0, value=5.0)
        pfua = st.number_input("PFUA (ng/mL)", min_value=0.0, max_value=100.0, value=0.5)

    # Create prediction button
    if st.button("Predict"):
        # Build input dataframe
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Race': [race],
            'Education': [education],
            'Smoke': [smoke],
            'Alcohol': [alcohol],
            'Age': [age],
            'BMI': [bmi],
            'PIR': [pir],
            'MPAH': [mpah],
            'PFDE': [pfde],
            'PFHxS': [pfhxs],
            'PFNA': [pfna],
            'PFOA': [pfoa],
            'PFOS': [pfos],
            'PFUA': [pfua]
        })

        # Standardize numerical features
        input_data[num_features] = scaler.transform(input_data[num_features])

        # Make prediction
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            # Display prediction results
            st.subheader("Prediction Results")

            # Get risk probability
            risk_probability = prediction_proba[0][1] * 100

            # Determine risk level
            if risk_probability < 30:
                risk_level = "Low Risk"
            elif risk_probability < 70:
                risk_level = "Medium Risk"
            else:
                risk_level = "High Risk"

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Prediction Result:**", 'Disease' if prediction[0] == 1 else 'Healthy')
                st.write("**Risk Level:**", risk_level)

            with col2:
                st.write(f"**Risk Probability:** {risk_probability:.2f}%")

            # Risk interpretation
            st.markdown("---")
            st.markdown("### Risk Interpretation")
            st.markdown("""
            - **Low Risk** (0-30%): Relatively low risk, maintain healthy lifestyle
            - **Medium Risk** (30-70%): Pay attention to health status, regular check-ups recommended
            - **High Risk** (70-100%): Immediate medical consultation recommended
            """)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    main()