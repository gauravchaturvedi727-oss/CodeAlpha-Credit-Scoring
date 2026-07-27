import joblib
import pandas as pd
model = joblib.load("models/credit_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_name = joblib.load("models/model_name.pkl")
encoders = joblib.load("models/encoders.pkl")

def predict_credit_score(
    age,
    sex,
    job,
    housing,
    saving_accounts,
    checking_account,
    credit_amount,
    duration,
    purpose
):

    data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }])
    categorical_cols = [
        "Sex",
        "Housing",
        "Saving accounts",
        "Checking account",
        "Purpose"
    ]

    for col in categorical_cols:

        encoder = encoders[col]

        value = str(data[col].iloc[0])

        if value not in encoder.classes_:
            value = encoder.classes_[0]

        data[col] = encoder.transform([value])

    if model_name == "Logistic Regression":
        data = scaler.transform(data)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].max()

    if prediction == 1:
        result = "GOOD CREDIT"
    else:
        result = "BAD CREDIT"

    return result, probability

if __name__ == "__main__":

    result, probability = predict_credit_score(
        age=35,
        sex="male",
        job=2,
        housing="own",
        saving_accounts="little",
        checking_account="moderate",
        credit_amount=2500,
        duration=24,
        purpose="radio/TV"
    )

    print(result)
    print(f"Probability : {probability:.2%}")