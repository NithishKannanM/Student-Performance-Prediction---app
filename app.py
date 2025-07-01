import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('best_student_performance_model.pkl')
st.title("Student Performance Prediction")

study_hours_per_day = st.slider("Study Hours per Day", 0, 12, 2)
attendance_percentage = st.slider("Attendance Percentage", 0, 100, 80) 
sleep_hours = st.slider("Sleep Hours", 0, 12, 8)
mental_health_rating = st.slider("Mental Health Rating", 1, 10, 5)
diet_quality = st.slider("Diet Quality (1-10)", 1, 10, 5)
part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
exercise_frequency = st.slider("Exercise Frequency (days per week)", 0, 7, 3)

ptj = 1 if part_time_job == "Yes" else 0

if st.button("Predict"):
    input_data = np.array([[study_hours_per_day, attendance_percentage, sleep_hours, mental_health_rating, diet_quality, ptj, exercise_frequency]])
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))  # Ensure prediction is within 0-100

    st.success(f"Predicted Score: {prediction:.2f}")