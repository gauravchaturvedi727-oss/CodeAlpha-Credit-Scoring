# 💳 Credit Scoring Model

A Machine Learning web application that predicts whether a customer has Good Credit or Bad Credit based on financial and personal information.

> ⚠️ Note: This project uses a demonstration dataset. Since the uploaded dataset did not contain a target label (`Risk`), a synthetic target was created only to demonstrate the end-to-end ML workflow.

---

## 📌 Features

- Data Preprocessing
- Missing Value Handling
- Label Encoding
- Feature Scaling
- Multiple ML Algorithms
- Automatic Best Model Selection
- Flask Web Application
- User Friendly Bootstrap Interface

---

## 🛠️ Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Bootstrap 5
- HTML
- CSS

---

## 📂 Project Structure

```
Credit-Scoring-Model/

│── app.py
│── requirements.txt
│── README.md

├── dataset/
│      german_credit_data.csv

├── models/
│      credit_model.pkl
│      scaler.pkl
│      model_name.pkl
│      encoders.pkl

├── src/
│      train.py
│      predict.py

├── templates/
│      index.html

├── static/
│      style.css
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Credit-Scoring-Model.git
```

Go inside the project

```bash
cd Credit-Scoring-Model
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
python src/train.py
```

Run Flask

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

## 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The application automatically selects the model with the highest accuracy.

---

## 📊 Input Features

- Age
- Sex
- Job
- Housing
- Saving Account
- Checking Account
- Credit Amount
- Loan Duration
- Purpose

---

## 📈 Output

- GOOD CREDIT
- BAD CREDIT

with prediction probability.

---

## 🔮 Future Improvements

- Use the original German Credit dataset with a real `Risk` target.
- Hyperparameter tuning.
- Model comparison dashboard.
- SHAP feature explanations.
- Deployment on Render.

---

## 👨‍💻 Author

**Gourav Chaturvedi**

---

⭐ If you like this project, give it a Star.
