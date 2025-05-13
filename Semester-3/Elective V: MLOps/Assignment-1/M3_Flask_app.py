from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the best-trained model (after Optuna tuning)
model = joblib.load('models/titanic_model_best.pkl')  # Updated to load the best model

# Define the home route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# API route for prediction using POST method (from form submission)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request has JSON or form data
        if request.is_json:
            # Get JSON data from the request
            data = request.get_json()
            
            # Extract the input features directly from JSON
            pclass = data['Pclass']
            sex = data['Sex']
            age = data['Age']
            sibsp = data['SibSp']
            parch = data['Parch']
            fare = data['Fare']
            embarked = data['Embarked']
            
            features = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                        columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
            
            # Make prediction using the model
            prediction = model.predict(features)
            
            # Map the prediction to a human-readable result (0: Not Survived, 1: Survived)
            result = 'Survived' if prediction[0] == 1 else 'Not Survived'
            
            # Return the prediction as a JSON response
            return jsonify({'prediction': result})

        else:
            # Extract form data for regular form submission
            pclass = int(request.form['Pclass'])
            sex = int(request.form['Sex'])
            age = float(request.form['Age'])
            sibsp = int(request.form['SibSp'])
            parch = int(request.form['Parch'])
            fare = float(request.form['Fare'])
            embarked = int(request.form['Embarked'])
            
            # Prepare the input feature array with column names
            features = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                                    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
            
            # Make a prediction
            prediction = model.predict(features)
            
            # Map the prediction to a human-readable result
            result = 'Survived' if prediction[0] == 1 else 'Not Survived'
            
            # Render the result in the form view
            return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        # Log and return the error
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
