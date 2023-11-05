from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name)

# Load the dataset
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Split the data into features (x) and target (y)
x = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

# Initialize and train the Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(x, y)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        age = float(request.form.get("age"))
        anaemia = int(request.form.get("anaemia"))
        creatinine_phosphokinase = int(request.form.get("creatinine_phosphokinase"))
        diabetes = int(request.form.get("diabetes"))
        ejection_fraction = int(request.form.get("ejection_fraction"))
        high_blood_pressure = int(request.form.get("high_blood_pressure"))
        serum_creatinine = float(request.form.get("serum_creatinine"))
        serum_sodium = int(request.form.get("serum_sodium"))
        sex = int(request.form.get("sex"))
        smoking = int(request.form.get("smoking"))
        time = int(request.form.get("time"))

        sample_data = [[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, serum_creatinine, serum_sodium, sex, smoking, time]]

        prediction = dt.predict(sample_data)
        return render_template("index.html", prediction=prediction[0])

    return render_template("index.html", prediction=None)

if __name__ == "__main":
    app.run(debug=True)
