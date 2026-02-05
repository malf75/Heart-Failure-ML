import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load('best_xgboost.pkl')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    age: int = Form(...),
    sex: str = Form(...),
    chestpain: str = Form(...),
    restingbp: int = Form(...),
    cholesterol: float = Form(...),
    fastingbs: int = Form(...),
    restingecg: str = Form(...),
    maxhr: int = Form(...),
    exerciseangina: str = Form(...),
    oldpeak: float = Form(...),
    stslope: str = Form(...)
):

    data = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'ChestPainType': chestpain,
        'RestingBP': restingbp,
        'Cholesterol': cholesterol,
        'FastingBS': fastingbs,
        'RestingECG': restingecg,
        'MaxHR': maxhr,
        'ExerciseAngina': exerciseangina,
        'Oldpeak': oldpeak,
        'ST_Slope': stslope
    }])

    proba = model.predict_proba(data)
    if proba[0][0] > proba[0][1]:
        result = f'{proba[0][0] * 100:.2f}% sem doença'
    else:
        result = f'{proba[0][1] * 100:.2f}% com doença'

    return JSONResponse({"result": result})
