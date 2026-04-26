from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('churn_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        recharge_amount = float(request.form['recharge_amount'])
        times_recharged = float(request.form['times_recharged'])

        # Validation
        if not (25 <= age <= 60):
            return render_template('index.html', prediction="Age should be between 25 and 60.")
        if not (150 <= recharge_amount <= 600):
            return render_template('index.html', prediction="Recharge amount should be between 150 and 600.")
        if times_recharged < 0:
            return render_template('index.html', prediction="Number of times recharged cannot be negative.")

        # Make prediction
        user_data = np.array([[age, recharge_amount, times_recharged]])
        prediction = model.predict(user_data)

        if prediction[0] == 1:
            result = "The customer is likely to churn."
        else:
            result = "The customer is not likely to churn."

        return render_template('index.html', prediction=result)
    except ValueError:
        return render_template('index.html', prediction="Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)