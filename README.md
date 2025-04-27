# Student Learning Progress Tracker

## Overview
This dashboard is a machine learning-powered tool designed to predict student performance on future questions, based on behavioral and performance data.

It helps identify trends in student learning progress, providing actionable insights for educators to better support their students' journeys.

## Features
- **Student Learning Curve**: Rolling accuracy plotted over time with a smoothed trendline for easy progress tracking.
- **Performance Prediction**: Predicts the likelihood of a student answering the next question correctly based on selected features like attempt number, difficulty level, and response time.
- **Clean, Professional Layout**: Side-by-side display of prediction results and model performance (ROC AUC Score) for quick and clear interpretation.

## Technical Details
- **Framework**: Streamlit
- **Model**: Random Forest Classifier
- **Data**: Simulated student-question dataset with behavioral features engineered (e.g., rolling accuracy, time since last attempt, average response time).
- **Evaluation**: Achieved ROC AUC score of 0.74, showing good predictive capability in a real-world educational context.

## Future Improvements
- Integrate real-time data streaming (live student responses).
- Add session-based engagement features (fatigue detection, time-of-day effects).
- Deploy on cloud infrastructure for scalability and broader access.

## Purpose
Built as part of a personal project to demonstrate machine learning application in education technology, aligning with the mission of improving student learning experiences through data-driven insights.

---

*Created by Maria — a passionate builder on a mission to support learning through technology.*
