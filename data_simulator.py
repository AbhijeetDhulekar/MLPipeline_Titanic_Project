import random
import time
import requests
import pandas as pd
from datetime import datetime
import os

# Configuration
API_URL = "http://127.0.0.1:8000/predict"  
DATA_STORAGE_PATH = "data/streaming_data"
os.makedirs(DATA_STORAGE_PATH, exist_ok=True)


def generate_random_passenger():
    sex = random.choice(["male", "female"])
    pclass = random.choice([1, 2, 3])
    age = random.randint(1, 80)
    sibsp = random.randint(0, 5)
    parch = random.randint(0, 4)
    fare = round(random.uniform(5, 200 if pclass == 1 else 100 if pclass == 2 else 50), 2)
    embarked = random.choice(["S", "C", "Q"])
    deck = random.choice(["A", "B", "C", "D", "E", "F", "G", "U"])
    
    
    survival_prob = 0.5
    if sex == "female": survival_prob += 0.3
    if pclass == 1: survival_prob += 0.2
    if age < 16: survival_prob += 0.1
    survived = int(random.random() < survival_prob)
    
    return {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Deck": deck,
        "Survived": survived  
    }

def send_to_api(passenger_data):
    try:
        # Remove ground truth before sending to API
        api_data = passenger_data.copy()
        api_data.pop("Survived", None)
        
        response = requests.post(API_URL, json=api_data)
        return response.json()
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

def store_passenger(passenger_data, prediction_result):
    # Add timestamp and prediction results
    record = passenger_data.copy()
    record.update({
        "timestamp": datetime.now().isoformat(),
        "predicted_survived": prediction_result.get("survived", None),
        "predicted_probability": prediction_result.get("probability", None)
    })
    
    # Save to daily batch file
    today = datetime.now().strftime("%Y-%m-%d")
    batch_file = f"{DATA_STORAGE_PATH}/{today}_data.csv"
    
    # Create or append to file
    df = pd.DataFrame([record])
    if os.path.exists(batch_file):
        df.to_csv(batch_file, mode='a', header=False, index=False)
    else:
        df.to_csv(batch_file, index=False)

def simulate_streaming(interval_seconds=60):
    """Simulate real-time data streaming"""
    while True:
        passenger = generate_random_passenger()
        print(f"Generated passenger: {passenger}")
        
        # Send to inference API
        prediction = send_to_api(passenger)
        if prediction:
            print(f"Prediction result: {prediction}")
            
            # Store with ground truth and prediction
            store_passenger(passenger, prediction)
        
        time.sleep(interval_seconds)

if __name__ == "__main__":
    simulate_streaming(interval_seconds=60)