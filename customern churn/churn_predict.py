## import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import json
import queue
import re
import sounddevice as sd
import pyttsx3
from vosk import Model, KaldiRecognizer
import os

# ============================================================
# LOAD DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(CSV_PATH)
pd.set_option('display.max_columns', None)

df = df.drop(columns=["customerID"])
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
df["churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
df = df.drop(columns=["Churn"])

# ============================================================
# DATA PREPROCESSING
# ============================================================

object_cols = df.select_dtypes(include=["object"]).columns.to_list()

encoders_dict = {}
for column in object_cols:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    encoders_dict[column] = le
    with open(f"encoder_{column}.pkl", "wb") as f:
        pickle.dump(le, f)

X = df.drop(columns=["churn"])
y = df["churn"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# ============================================================
# MODEL TRAINING
# ============================================================

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
}

print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

for name, model in models.items():
    scores = cross_val_score(model, x_train_smote, y_train_smote, cv=5)
    print(f"{name:20} - Accuracy: {scores.mean():.4f}")

# ============================================================
# 🔥 FINAL MODEL (ACCURACY IMPROVED – TUNED RF)
# ============================================================

rfc = RandomForestClassifier(
    n_estimators=400,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rfc.fit(x_train_smote, y_train_smote)

y_test_pred = rfc.predict(x_test)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

model_data = {
    "model": rfc,
    "features_names": X.columns.tolist(),
    "categorical_columns": object_cols
}

with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]
categorical_columns = model_data["categorical_columns"]

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_customer_churn(customer_data_dict):
    input_df = pd.DataFrame([customer_data_dict])

    encoders_loaded = {}
    for col in categorical_columns:
        with open(f"encoder_{col}.pkl", "rb") as f:
            encoders_loaded[col] = pickle.load(f)

    for col in categorical_columns:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            input_df[col] = input_df[col].apply(
                lambda x: x if x in encoders_loaded[col].classes_
                else encoders_loaded[col].classes_[0]
            )
            input_df[col] = encoders_loaded[col].transform(input_df[col])

    input_df = input_df[feature_names]

    pred = loaded_model.predict(input_df)[0]
    prob = loaded_model.predict_proba(input_df)[0]

    return {
        "prediction": "Churn" if pred == 1 else "No Churn",
        "churn_probability": prob[1],
        "no_churn_probability": prob[0]
    }

# ============================================================
# VOICE SYSTEM (UNCHANGED)
# ============================================================

tts = pyttsx3.init()
tts.setProperty("rate", 170)

def speak(text):
    print("🤖:", text)
    tts.say(text)
    tts.runAndWait()

VOSK_MODEL_PATH = r"C:\Users\ABISH\Desktop\python\models\vosk_en"
vosk_model = Model(VOSK_MODEL_PATH)

def listen(seconds=6):
    q = queue.Queue()
    samplerate = 16000
    rec = KaldiRecognizer(vosk_model, samplerate)

    def callback(indata, frames, time, status):
        q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback
    ):
        for _ in range(int(seconds * samplerate / 8000)):
            rec.AcceptWaveform(q.get())

    return json.loads(rec.FinalResult()).get("text", "")

if __name__ == "__main__":
    print("✅ System ready. Model + Voice assistant loaded.")

# ==================== EXAMPLE PREDICTIONS ====================

print("\n" + "="*60)
print("MAKING PREDICTIONS ON NEW CUSTOMERS")
print("="*60)

# Example 1: New customer (high churn risk)
example_customer_1 = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

result_1 = predict_customer_churn(example_customer_1)
if result_1:
    print(f"\n🔍 CUSTOMER 1 (New Budget Subscriber):")
    print(f"   Prediction: {result_1['prediction']}")
    print(f"   Churn Probability: {result_1['churn_probability']:.1%}")
    print(f"   No Churn Probability: {result_1['no_churn_probability']:.1%}")

# Example 2: Long-term customer (low churn risk)
example_customer_2 = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 36,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 119.85,
    "TotalCharges": 4314.60
}

result_2 = predict_customer_churn(example_customer_2)
if result_2:
    print(f"\n🔍 CUSTOMER 2 (Long-term Premium Subscriber):")
    print(f"   Prediction: {result_2['prediction']}")
    print(f"   Churn Probability: {result_2['churn_probability']:.1%}")
    print(f"   No Churn Probability: {result_2['no_churn_probability']:.1%}")

print("\n" + "="*60)
print("✅ PREDICTIONS COMPLETE")
print("="*60)
