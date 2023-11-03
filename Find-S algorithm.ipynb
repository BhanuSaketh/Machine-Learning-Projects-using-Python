import pandas as pd

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

print(df)

h = ['@'] * (df.shape[1] - 1)
print("H 0:")
print(h)

for i, value in enumerate(df['EnjoySport']):
    if value == 'Yes':
        h = df.iloc[i][:-1].tolist()
        print(f"H {i + 1}:")
        print(h)
        break
    else:
        print(f"H {i + 1} Ignored")

for j in range(i + 1, len(df)):
    if df.iloc[j][-1] == 'Yes':
        for k in range(len(h)):
            if h[k] != df.iloc[j][k] and h[k] != '?':
                h[k] = '?'
        print(f"H {j + 1}:")
        print(h)
    else:
        print(f"H {j + 1} Ignored")

# Model Prediction
h1 = ['Sunny', 'Warm', 'Normal', 'Strong', 'Cool', 'Same']
flag = all(a == '?' or a == b for a, b in zip(h, h1))
print(f"Target Variable: {'Yes' if flag else 'No'}")
