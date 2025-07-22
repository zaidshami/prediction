from fastapi import FastAPI
import pandas as pd
import xgboost as xgb
import pickle

app = FastAPI()

# تحميل النموذج
model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def root():
    return {"status": "Model is running"}

@app.post("/predict/")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return {
            "prediction": float(prediction[0]),
            "probability": float(prediction[0])  # تُستخدم نفس القيمة كمؤشر للاتجاه
        }
    except Exception as e:
        return {"error": str(e)}
