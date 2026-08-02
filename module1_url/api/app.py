from fastapi import FastAPI
from pydantic import BaseModel
from models.predict import predict_url

app = FastAPI(title="Website URL Phishing Detection")


class URLRequest(BaseModel):
    url: str


@app.get("/")
def root():
    return {"message": "Website URL Phishing Detection API"}


@app.post("/predict")
def predict(data: URLRequest):
    return predict_url(data.url)
