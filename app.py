from flask import Flask, render_template, request
from src.predict import predict_credit_score

app = Flask(__name__)

app.config["SECRET_KEY"] = "credit_scoring_project"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        age = int(request.form["age"])
        sex = request.form["sex"]
        job = int(request.form["job"])
        housing = request.form["housing"]
        saving_accounts = request.form["saving_accounts"]
        checking_account = request.form["checking_account"]
        credit_amount = float(request.form["credit_amount"])
        duration = int(request.form["duration"])
        purpose = request.form["purpose"]

        result, probability = predict_credit_score(
            age,
            sex,
            job,
            housing,
            saving_accounts,
            checking_account,
            credit_amount,
            duration,
            purpose
        )

        return render_template(
            "index.html",
            prediction=result,
            probability=round(probability * 100, 2)
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction="Error",
            probability=0,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)