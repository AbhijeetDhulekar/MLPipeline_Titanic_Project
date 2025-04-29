import time
from datetime import datetime, timedelta
from model_registry import register_new_version

RETRAIN_INTERVAL_SECONDS = 30  

def retrain_loop():
    last_run = None
    
    while True:
        now = datetime.now()
        
        # Check if it's time to retrain
        if last_run is None or (now - last_run) > timedelta(seconds=RETRAIN_INTERVAL_SECONDS):
            print(f"{now}: Starting retraining...")
            try:
                version_info = register_new_version()
                if version_info:
                    print(f"Retraining completed. Version {version_info['version_id']}")
                    print(f"Metrics: {version_info['metrics']}")
                last_run = now
            except Exception as e:
                print(f"Retraining failed: {str(e)}")
        
        time.sleep(5)  # Check every 5 seconds (smaller sleep for more precise timing)

if __name__ == "__main__":
    retrain_loop()