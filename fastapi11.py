from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI app
app = FastAPI()

# Load your trained model (.pkl file)
model = joblib.load("heart_disease_model.pkl")

# Define input data format
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/version")
def version():
    return {"api_version": "1.0.0"}

# Create prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    # Prepare the input data in correct order
    input_data = [[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]]
    
    # Make prediction
    prediction = model.predict(input_data)[0]

    # Map prediction to a readable result
    if prediction == 1:
        result = "Heart Disease Likely"
    else:
        result = "No Heart Disease"

    # Return the prediction result as JSON
    return {
        "prediction": int(prediction),
        "result": result
    }
