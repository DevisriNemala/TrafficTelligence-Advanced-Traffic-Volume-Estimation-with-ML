import numpy as np
import pandas as pd
import joblib  # ✅ using joblib instead of pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# ✅ Load compressed model
pipeline = joblib.load("model_pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        holiday = request.form["holiday"]
        temp = float(request.form["temp"])
        rain = float(request.form["rain"])
        snow = float(request.form["snow"])
        weather = request.form["weather"]

        input_data = pd.DataFrame([{
            "holiday": holiday,
            "temp": temp,
            "rain": rain,
            "snow": snow,
            "weather": weather
        }])

        prediction = pipeline.predict(input_data)[0]
        prediction = round(prediction)

        if prediction > 5000:
            return render_template("chance.html", prediction_text=f"Estimated Traffic Volume: {prediction}")
        else:
            return render_template("noChance.html", prediction_text=f"Estimated Traffic Volume: {prediction}")

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
