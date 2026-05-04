import numpy as np
from flask import Flask, render_template, request
import pickle

# CREATE APP FIRST (VERY IMPORTANT)
app = Flask(__name__)

# Load model
with open("Model.pkl", "rb") as f:
    m, scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # take numeric input
        city = int(data['city'])
        country = int(data['country'])

        a = [
            float(data['bedrooms']),
            float(data['bathrooms']),
            float(data['sqft_living']),
            float(data['sqft_lot']),
            float(data['floors']),
            float(data['waterfront']),
            float(data['view']),
            float(data['condition']),
            float(data['sqft_above']),
            float(data['sqft_basement']),
            float(data['yr_built']),
            float(data['yr_renovated']),
            city,
            country
        ]

        b = np.array([a])
        b = scaler.transform(b)

        prediction = m.predict(b)[0]

        return render_template("index.html", prediction_text=prediction)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)