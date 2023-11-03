# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

""" Link for dataset  
                      https://www.kaggle.com/datasets/zhaoyingzhu/heartcsv """

# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Heart.csv")
data = data.drop("Unnamed: 0", axis=1)

# Encode categorical variables
label_encoder = LabelEncoder()
data["ChestPain"] = label_encoder.fit_transform(data["ChestPain"])
data['Thal'] = label_encoder.fit_transform(data["Thal"])
data['AHD'] = label_encoder.fit_transform(data["AHD"])

# Drop rows with missing values
data = data.dropna()

# Split the dataset into features (x) and target (y)
x = data.drop("AHD", axis=1)
y = data["AHD"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create and train a logistic regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Make predictions on the training data
train_predictions = lr.predict(x_train)

# Calculate the accuracy on the training data
train_accuracy = lr.score(x_train, y_train)
print("Training Accuracy:", train_accuracy)

# Make predictions on the test data
test_predictions = lr.predict(x_test)

# Calculate the accuracy on the test data
test_accuracy = lr.score(x_test, y_test)
print("Test Accuracy:", test_accuracy)
