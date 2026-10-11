from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("spam_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    message = data.get("message", "")

    if not message:
        return jsonify({"error": "Please enter a message"}), 400

    prediction = model.predict([message])[0]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    return jsonify({
        "message": message,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)