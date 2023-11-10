import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

"""Dataset link   https://www.kaggle.com/datasets/jeevanrh/drug200csv  """


data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/drug200.csv")

# Encoding categorical variables
encoder = LabelEncoder()
data['BP'] = encoder.fit_transform(data['BP'])
data['Sex'] = encoder.fit_transform(data['Sex'])
data['Cholesterol'] = encoder.fit_transform(data['Cholesterol'])
data['Drug'] = encoder.fit_transform(data['Drug'])

x = data.drop('Drug', axis=1)
y = data['Drug']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)

model = GaussianNB()
model.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print("Training accuracy:", train_score)
print("Test accuracy:", test_score)

# Test data for prediction (reshape as per the number of features)
test = np.array([20, 0, 2, 0, 7.798]).reshape(1, -1)
prediction = model.predict(test)
print("Predicted drug:", encoder.inverse_transform(prediction))
