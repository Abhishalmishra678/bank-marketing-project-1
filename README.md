# Bank marketing project 
## Overview
This project involves building a machine learning model to analyze bank data provided by the organization. The main goal is to predict customer behavior based on the provided dataset.

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- joblib

## Project Structure
import pickle
downlod the model from the same repository.
# Load the model
with open('models/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions
predictions = model.predict(test_data)
