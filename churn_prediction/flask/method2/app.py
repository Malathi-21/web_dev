from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

app = Flask(__name__, template_folder='templates')

# Example dataset (replace with real data if available)
import pandas as pd
np.random.seed(123)
df = pd.read_csv("prep.csv")

# Prepare training data
X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'OnlineSecurity_Yes', 'Contract_Two_year']]
y = df['Churn_Yes']

# Feature selection and scaling
selector = SelectKBest(score_func=chi2, k=X.shape[1])
X_selected = selector.fit_transform(X, y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        tenure = int(request.form['tenure'])
        MonthlyCharges = float(request.form['MonthlyCharges'])
        TotalCharges = float(request.form['TotalCharges'])
        OnlineSecurity_Yes = float(request.form['OnlineSecurity_Yes'])
        Contract_Two_year = float(request.form['Contract_Two_year'])

        # Prepare input for prediction
        input_data = np.array([[tenure, MonthlyCharges, TotalCharges, OnlineSecurity_Yes, Contract_Two_year]])
        input_selected = selector.transform(input_data)  # Apply the same feature selection
        input_scaled = scaler.transform(input_selected)  # Apply the same scaling

        # Make prediction
        prediction = model.predict(input_scaled)
        result = 'Churn Predicted' if prediction[0] == 1 else 'No Churn'

        return render_template('output.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
