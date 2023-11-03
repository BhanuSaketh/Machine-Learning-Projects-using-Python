import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

%matplotlib inline

# Load the Iris dataset
data = sns.load_dataset("iris")


# Create a scatter plot to visualize the data
sns.scatterplot(data=data, x="petal_length", y="petal_width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Scatter Plot of Petal Length vs. Petal Width")
plt.show()

# Extract relevant data columns
data = data[['petal_width', 'petal_length']]

# Split data into training and testing sets
x = data["petal_length"].values.reshape(-1, 1)
y = data["petal_width"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=23)

# Create and train the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Get the model parameters
intercept = lr.intercept_
coef = lr.coef_[0]

# Predict on the training data
y_pred_train = lr.predict(x_train)

# Plot the training data and regression line
plt.scatter(x_train, y_train, label="Training Data")
plt.plot(x_train, y_pred_train, color='red', label="Regression Line")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear Regression on Training Data")
plt.legend()
plt.show()

# Predict on the testing data
y_pred_test = lr.predict(x_test)

# Plot the testing data and regression line
plt.scatter(x_test, y_test, label="Testing Data")
plt.plot(x_test, y_pred_test, color='red', label="Regression Line")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear Regression on Testing Data")
plt.legend()
plt.show()

print("Intercept (c):", intercept)
print("Coefficient (m):", coef)
