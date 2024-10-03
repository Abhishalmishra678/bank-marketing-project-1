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


## Steps Involved
1. **Data Import**: Loaded the training and testing datasets using Pandas.
2. **Data Cleaning**: Handled null values and performed data transformations to prepare the data for analysis.
3. **Data Analysis**: Analyzed the data using various statistical techniques and visualizations.
4. **Handling Skewness and Outliers**: Used appropriate methods to treat skewness and outliers in the dataset.
5. **Graphical Analysis**: Visualized data distributions and relationships using Matplotlib and Seaborn.
6. **Data Scaling**: Scaled the features using StandardScaler.
7. **Train-Test Split**: Split the dataset into training and testing sets using the `train_test_split` function.
8. **Model Training**: Evaluated multiple machine learning models, with Random Forest Classifier proving to be the best through cross-validation.
9. **Model Testing**: Tested the model on the provided testing dataset.
10. **Model Saving**: Saved the trained model for future use.

## Model Performance
The Random Forest Classifier was selected as the best model based on cross-validation results. It demonstrated high accuracy in predicting customer behavior on the testing dataset.


## dataset link
https://raw.githubusercontent.com/dsrscientist/dataset5/main/termdeposit_train.csv
https://raw.githubusercontent.com/dsrscientist/dataset5/main/termdeposit_test.csv


## Usage
To use the saved model, load it using the following code snippet:

```python
import pickle

# Load the model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions
predictions = model.predict(test_data)
