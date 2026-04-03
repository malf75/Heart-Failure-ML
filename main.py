import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import StandardScaler

app = FastAPI()

templates = Jinja2Templates(directory="templates")

stacking_model = joblib.load('./stacking.pkl')

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
        'ST_Slope': stslope,
        'Diff_Age_MaxHR': maxhr - age,
        'Bi_ExerciseAngina_ST_Slope': exerciseangina + '_' + stslope,
        'Bi_ExerciseAngina_ChestPainType': exerciseangina + '_' + chestpain
    }])

    data[['Age_scaled', 'Oldpeak_scaled']] = StandardScaler().fit_transform(data[['Age', 'Oldpeak']])
    data['Age_Oldpeak_Sum'] = data['Age_scaled'] + data['Oldpeak_scaled']

    data.drop(['Age_scaled', 'Oldpeak_scaled'], axis=1, inplace=True)
    prob = stacking_model.predict_proba(data)[0]
    
    if prob[1] > 0.5:
        result = f"{prob[1] * 100:.2f}% com doença"
    else:
        result = f"{prob[0] * 100:.2f}% sem doença"

    return JSONResponse({"result": result})
