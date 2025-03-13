# app.py
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["message"]
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template("index.html", result=result, message=user_input)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
