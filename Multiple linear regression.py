import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

""" link for dataset 
                    https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression     """

# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Student_Performance.csv")

# Encode the 'Extracurricular Activities' column
data["Extracurricular Activities"] = LabelEncoder().fit_transform(data["Extracurricular Activities"])

# Split the data into features (x) and target (y)
x = data.drop("Performance Index", axis=1)
y = data["Performance Index"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Create and train the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Get the intercept and coefficients
c = lr.intercept_
m = lr.coef_

# Print the intercept and coefficients
print("Intercept (C):", c)
print("Coefficients (m):", m)

# Predict on the training data
y_pred_train = lr.predict(x_train)

# Plot actual vs. predicted on the training data
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train, color='blue', label="Actual vs. Predicted (Training Data)")
plt.xlabel("Actual Performance Index (Training Data)")
plt.ylabel("Predicted Performance Index (Training Data)")
plt.title("Actual vs. Predicted Performance Index (Training Data)")
plt.legend()
plt.grid(True)
plt.show()

# Calculate the R-squared score on the training data
r2_train = r2_score(y_train, y_pred_train)
print("R-squared (Training Data):", r2_train)

# Predict on the test data
y_pred_test = lr.predict(x_test)

# Plot actual vs. predicted on the test data
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='green', label="Actual vs. Predicted (Test Data)")
plt.xlabel("Actual Performance Index (Test Data)")
plt.ylabel("Predicted Performance Index (Test Data)")
plt.title("Actual vs. Predicted Performance Index (Test Data)")
plt.legend()
plt.grid(True)
plt.show()

# Calculate the R-squared score on the test data
r2_test = r2_score(y_test, y_pred_test)
print("R-squared (Test Data):", r2_test)
