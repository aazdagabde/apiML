from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import os
from pydantic import BaseModel

app = FastAPI()

# Définition des chemins vers les modèles
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
WEATHER_MODEL_PATH = os.path.join(MODEL_DIR, 'weather_model.pkl')
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')

# Chargement du modèle météo
with open(WEATHER_MODEL_PATH, 'rb') as f:
    weather_model = pickle.load(f)

# Chargement du modèle d'anomalie
with open(ANOMALY_MODEL_PATH, 'rb') as f:
    anomaly_data = pickle.load(f)
    # On suppose que le fichier pickle contient un dictionnaire avec 'model' et 'scaler'
    anomaly_model = anomaly_data.get('model')
    scaler = anomaly_data.get('scaler')

class SensorData(BaseModel):
    temperature: float
    humidite: float
    pression: float
    qualite_air: float

@app.post("/predict")
def predict(data: SensorData):
    # Construction de la donnée pour les prédictions
    X = np.array([[data.temperature, data.humidite, data.pression, data.qualite_air]])
    
    # Prédiction de pluie avec le modèle météo
    try:
        pluie = bool(weather_model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction météo: {e}")
    
    # Détection d'anomalie avec le modèle d'anomalie (après scaling)
    try:
        X_scaled = scaler.transform(X)
        anomalie = bool(anomaly_model.predict(X_scaled)[0] == -1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la détection d'anomalie: {e}")

    return {"pluie": pluie, "anomalie": anomalie}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
