from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# =========================
# Load Saved Files
# =========================
model = pickle.load(open('model_knn.save', 'rb'))
scaler = pickle.load(open('scaled_knn.save', 'rb'))

# Convert model columns into list
modelcolumns = list(pickle.load(open('model_columns.pk1', 'rb')))

# Remove target column if exists
if 'HeartDisease' in modelcolumns:
    modelcolumns.remove('HeartDisease')


# =========================
# Home Page
# =========================
@app.route('/')
def home():
    return render_template('index.html')


# =========================
# Prediction Route
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    # Numerical Inputs
    Age = float(request.form['Age'])
    RestingBP = float(request.form['RestingBP'])
    Cholesterol = float(request.form['Cholesterol'])
    FastingBS = float(request.form['FastingBS'])
    MaxHR = float(request.form['MaxHR'])
    Oldpeak = float(request.form['Oldpeak'])

    # Categorical Inputs
    Sex = request.form['Sex']
    ChestpainType = request.form['ChestpainType']
    RestingECG = request.form['RestingECG']
    ExerciseAngina = request.form['ExerciseAngina']
    ST_Slope = request.form['ST_Slope']

    # =========================
    # Binary Encoding
    # =========================
    Sex = 1 if Sex == "Male" else 0
    ExerciseAngina = 1 if ExerciseAngina == "Y" else 0

    # =========================
    # Create Input DataFrame
    # =========================
    input_data = pd.DataFrame([{
        'Age': Age,
        'Sex': Sex,
        'ChestpainType': ChestpainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }])

    # =========================
    # Convert categorical columns
    # =========================
    input_df = pd.get_dummies(input_data)

    # =========================
    # Add Missing Columns
    # =========================
    for col in modelcolumns:
        if col not in input_df.columns:
            input_df[col] = 0

    # =========================
    # Keep Same Column Order
    # =========================
    input_df = input_df[modelcolumns]

    # =========================
    # Scale Features
    # =========================
    scaled_features = scaler.transform(input_df)

    # =========================
    # Prediction
    # =========================
    prediction = model.predict(scaled_features)

    # =========================
    # Result
    # =========================
    if prediction[0] == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return render_template('index.html', prediction_text=result)


# =========================
# Run Flask App
# =========================
if __name__ == "__main__":
    app.run(debug=True)