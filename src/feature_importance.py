import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

df = pd.read_csv("dataset/german_credit_data.csv")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)
    
df["Saving accounts"].fillna("Unknown", inplace=True)
df["Checking account"].fillna("Unknown", inplace=True)

df["Risk"] = (
    (
        (df["Checking account"] == "rich")
        |
        (df["Saving accounts"] == "rich")
    )
).astype(int)

encoders = {}

for col in df.select_dtypes(include="object").columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    encoders[col] = encoder

X = df.drop("Risk", axis=1)

y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

}


best_model = None
best_name = ""
best_accuracy = 0

for name, model in models.items():

    if name == "Logistic Regression":

        model.fit(X_train_scaled, y_train)

        prediction = model.predict(X_test_scaled)

    else:

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print("-" * 50)
    print(name)
    print("Accuracy :", accuracy)

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_name = name

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/credit_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")

joblib.dump(best_name, "models/model_name.pkl")

joblib.dump(encoders, "models/encoders.pkl")


print("\nBest Model :", best_name)
print("Accuracy :", best_accuracy)

print("\nTraining Completed Successfully.")