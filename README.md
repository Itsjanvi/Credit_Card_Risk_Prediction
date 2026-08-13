from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)


# Load trained model
model = joblib.load("model/credit_risk_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from form

        age = float(request.form["age"])
        income = float(request.form["income"])
        credit_limit = float(request.form["credit_limit"])
        utilization = float(request.form["utilization"])
        late_payments = float(request.form["late_payments"])
        outstanding_balance = float(request.form["outstanding_balance"])
        loan_amount = float(request.form["loan_amount"])
        credit_history = float(request.form["credit_history"])
        employment_years = float(request.form["employment_years"])


        # Create input array

        features = np.array([[
            age,
            income,
            credit_limit,
            utilization,
            late_payments,
            outstanding_balance,
            loan_amount,
            credit_history,
            employment_years
        ]])


        # Prediction

        prediction = model.predict(features)[0]

        probability = model.predict_proba(features)[0]

        high_risk_probability = probability[1] * 100


        if prediction == 1:
            result = "HIGH RISK"
            message = "The customer has a higher credit risk."
        else:
            result = "LOW RISK"
            message = "The customer has a lower credit risk."


        return render_template(
            "index.html",
            prediction=result,
            message=message,
            probability=round(high_risk_probability, 2)
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction="Error",
            message=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)