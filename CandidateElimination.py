import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

label_encoders = {}
categorical_columns = ['sky', 'airTemp', 'humidity', 'wind', 'water', 'forecast', 'enjoySport']
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

#candidateElimination Algorithm


def initialize_g(d):
    # Initialize the general boundary set to the most general hypothesis
    g = np.array(['?'] * d)
    return g
def initialize_s(d):
    # Initialize the specific boundary set to the most specific hypothesis
    s = np.array(['0'] * d)
    return s
def is_consistent(hypothesis, example):
    for h, e in zip(hypothesis, example):
        if h != '?' and h != e:
            return False
    return True
def candidate_elimination(examples):
    d = len(examples.columns) - 1  # Number of features (excluding the target column)
    
    G = [initialize_g(d)]
    S = [initialize_s(d)]

    for index, row in examples.iterrows():
        x = row[:-1].tolist()  # Features
        y = row[-1]  # Target label

        if y == 1:  # 'Yes' label
            G = [g for g in G if is_consistent(g, x)]

            s = S[0].copy()
            for i in range(d):
                if s[i] == '0':
                    s[i] = x[i]
                elif s[i] != x[i]:
                    s[i] = '?'
            S = [s]
        else:  # 'No' label
            S = [s for s in S if not is_consistent(s, x)]

            G = []
            for s in S:
                g = s.copy()
                for i in range(d):
                    if s[i] != x[i] and g[i] != '?':
                        g[i] = '?'
                G.append(g)

    return G, S

G, S = candidate_elimination(data)

print("General Boundary Set (G):")
for g in G:
    print(g)

print("\nSpecific Boundary Set (S):")
for s in S:
    print(s)
