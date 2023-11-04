import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
""" Lin for dataset 
                      https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data      """
# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/heart_failure_clinical_records_dataset.csv")

# Check for missing values
missing_values = data.isna().sum()
print("Missing Values:\n", missing_values)

# Split the data into features (x) and target (y)
x = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=44)

# Initialize and train the Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# Evaluate the model on the training data
train_accuracy = dt.score(x_train, y_train)
print("Training Accuracy:", train_accuracy)

# Predict using the trained model
sample_data = [[65.0, 0, 146, 0, 20, 1, 162000.00, 1.3, 129, 1, 1, 7]]
predictions = dt.predict(sample_data)
print("Predicted Outcome:", predictions)
