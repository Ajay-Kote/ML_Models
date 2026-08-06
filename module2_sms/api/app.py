from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="SMS Smishing Detection API")

MODEL_DIR = "model/saved_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

class SMSRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "SMS Smishing Detection API running"}

@app.post("/predict")
def predict(request: SMSRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "smishing" if pred == 1 else "legit"
    return {
        "text": request.text,
        "prediction": label,
        "confidence": round(confidence, 4)
    }