# Culinary Intelligence Dashboard

This project was developed as part of my Machine Learning Internship with Cognifyz Technologies. It demonstrates how machine learning can be applied to restaurant analytics, combining data modeling, visualization, and a functional web-based dashboard.

---

## Project Overview

The goal of this project is to analyze restaurant data and build machine learning models that can help users make informed decisions. From predicting ratings to recommending similar restaurants, this project brings together multiple real-world tasks under one dashboard.

---

## Objectives

1. **Predict Restaurant Ratings**  
   A regression model trained to estimate ratings based on location, cost, cuisine, and user feedback data.

2. **Recommend Similar Restaurants**  
   A content-based recommendation system that uses restaurant features to suggest similar dining options.

3. **Classify Cuisine Type**  
   A classification model that predicts the type of cuisine offered based on user and restaurant attributes.

4. **Location-Based Analysis**  
   An interactive map visualization that displays restaurant locations using latitude and longitude data.

---

## Technologies Used

- Python 3.10
- Flask (for building the web application)
- Jupyter Notebook (for prototyping and model development)
- pandas, numpy (data processing)
- scikit-learn (machine learning)
- folium (geospatial mapping)
- HTML/CSS with Bootstrap (for front-end design)

---

## File Structure
CulinaryDashboard/
├── app.py # Main Flask application
├── ML_Tasks.py # ML models and processing logic
├── ML_Tasks.ipynb # Jupyter notebook for analysis
├── Dataset .csv # Source dataset
├── requirements.txt # List of Python dependencies
├── templates/
│ ├── index.html
│ ├── result.html
│ └── map.html
├── static/
│ └── Restaurant_Location_Map.html
└── README.md


---

## How to Run This Project Locally

1. Clone the repository:
git clone https://github.com/tanushreedhananjayb/CulinaryDashboard.git
cd CulinaryDashboard


2. Set up a virtual environment:

python -m venv venv
source venv/bin/activate # For Windows: venv\Scripts\activate


3. Install the required dependencies:

pip install -r requirements.txt


4. Launch the application:

python app.py


5. Open your browser and navigate to:

http://127.0.0.1:5000


---

## Key Features and Results

- High accuracy achieved in rating prediction using Random Forest Regressor
- Effective recommendations generated using cosine similarity and feature vectors
- Cuisine classification using logistic regression shows promising results
- Folium-based map clearly visualizes restaurant distributions across cities

---

## Learning Outcomes

- Developed practical skills in machine learning model building and deployment
- Improved ability to modularize code using Python scripts and Jupyter notebooks
- Gained hands-on experience with Flask for building interactive dashboards
- Learned optimization strategies for deploying lightweight models and apps

---

## Project Status

All tasks have been successfully completed and validated. The code has been modularized and documented for reuse. A final submission video and zip file have been prepared as part of the internship requirements.

---

## Acknowledgments

Special thanks to Cognifyz Technologies for providing the opportunity to work on this problem statement. Gratitude to the mentors and evaluators who supported the process through feedback and guidance.

---

## Contact

**Tanushree Dhananjay Bhamare**  
Email: tanushreebhamare19@gmail.com 
GitHub: [https://github.com/tanushreedhananjayb](https://github.com/tanushreedhananjayb)  
LinkedIn: [https://www.linkedin.com/in/tanushreedhananjayb](https://www.linkedin.com/in/tanushreedhananjayb)

---

This project is part of the Machine Learning Internship Program at Cognifyz Technologies.




