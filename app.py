from flask import Flask, render_template, request
import numpy as np
import pickle

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("model/crop_model.keras")

# Load scaler
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load label encoder
with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    features = np.array([
        [N, P, K, temperature, humidity, ph, rainfall]
    ])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    predicted_index = np.argmax(prediction, axis=1)

    predicted_crop = le.inverse_transform(predicted_index)[0]

    return render_template(
        'result.html',
        prediction=predicted_crop
    )


if __name__ == '__main__':
    app.run(debug=True)