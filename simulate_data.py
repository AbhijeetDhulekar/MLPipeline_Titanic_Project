import pandas as pd
import requests
import time
import random

df = pd.read_csv('titanic.csv')

while True:
    sample = df.sample(1).to_dict(orient='records')[0]
    try:
        response = requests.post("http://localhost:8000/predict", json=sample)
        print(f"Sent: {sample} | Prediction: {response.json()}")
        
        # Store for retraining
        with open("new_data.csv", "a") as f:
            f.write(f"{sample}\n")
            
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(60)  # Send every minute