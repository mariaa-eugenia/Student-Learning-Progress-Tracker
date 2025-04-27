import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load your model and your data
model = joblib.load('student_model.pkl')  # <-- Save the model first if needed
df = pd.read_csv('student_data.csv', parse_dates=['timestamp']) # <-- student-question dataset


# Streamlit app
st.title("Student Learning Progress Tracker")

st.sidebar.header("Choose Student Features")
attempt_number = st.sidebar.slider('Attempt Number', 1, 5, 1)
difficulty_level = st.sidebar.slider('Difficulty Level', 1, 5, 3)
response_time = st.sidebar.slider('Response Time (seconds)', 5, 120, 30)
rolling_accuracy_last_5 = st.sidebar.slider('Rolling Accuracy (0 to 1)', 0.0, 1.0, 0.5)
time_since_last_attempt = st.sidebar.slider('Time Since Last Attempt (minutes)', 0, 500, 60)
question_repeat_count = st.sidebar.slider('Question Repeat Count', 0, 5, 0)
student_avg_response_time = st.sidebar.slider('Student Avg Response Time (seconds)', 5, 120, 30)

#
st.subheader("Student Learning Curve (Rolling Accuracy Over Time)")

# Let user pick a student
selected_student = st.selectbox("Select a Student ID:", df['student_id'].unique())

# Filter the data
student_data = df[df['student_id'] == selected_student].sort_values('timestamp')

# First calculate rolling_accuracy (over last 5 questions)
student_data['rolling_accuracy'] = student_data['is_correct'].rolling(window=5, min_periods=1).mean()

# Then create a smooth version (moving average of rolling_accuracy)
student_data['smooth_accuracy'] = student_data['rolling_accuracy'].rolling(window=3, min_periods=1).mean()


fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(student_data['timestamp'], student_data['rolling_accuracy'], marker='o', label='Rolling Accuracy', color='blue')
ax.plot(student_data['timestamp'], student_data['smooth_accuracy'], linestyle='--', color='red', label='Trendline')

ax.set_title(f"Rolling Accuracy for {selected_student}")
ax.set_xlabel("Time")
ax.set_ylabel("Rolling Accuracy (Last 5)")
ax.grid(True)
ax.legend()

# Format x-axis
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Prediction input
input_data = pd.DataFrame({
    'attempt_number': [attempt_number],
    'difficulty_level': [difficulty_level],
    'response_time': [response_time],
    'rolling_accuracy_last_5': [rolling_accuracy_last_5],
    'time_since_last_attempt': [time_since_last_attempt],
    'question_repeat_count': [question_repeat_count],
    'student_avg_response_time': [student_avg_response_time]
})

# Make prediction
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0][1]

# Display prediction
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"Likely Correct! (Confidence: {prediction_proba:.2f})")
    else:
        st.error(f"Likely Incorrect! (Confidence: {1 - prediction_proba:.2f})")

with col2:
    st.subheader("Model Info")
    st.metric(label="ROC AUC Score", value="0.74")
