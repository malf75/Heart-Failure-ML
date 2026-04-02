import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model_xgb = joblib.load('best_xgboost.pkl')
model_rf = joblib.load('best_random_forest.pkl')
model_lr = joblib.load('best_logistic_regression.pkl')
model_svm = joblib.load('best_svm.pkl')
model_nn = joblib.load('best_nn.pkl')

models = [model_xgb, model_rf, model_lr, model_svm, model_nn]

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

    probs = [m.predict_proba(data)[0] for m in models]
    model_names = ["XGBoost", "Random Forest", "Regressão Logística", "SVM", "Rede Neural"]
    results = {}
    for name, p in zip(model_names, probs):
        if p[1] > 0.5:
            results[name] = f"{p[1] * 100:.2f}% com doença"
        else:
            results[name] = f"{p[0] * 100:.2f}% sem doença"

    return JSONResponse({"results": results})
