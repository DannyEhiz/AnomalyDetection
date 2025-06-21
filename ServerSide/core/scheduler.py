from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from ServerSide.services.modelling import train_models 
import subprocess


scheduler = BlockingScheduler()

@scheduler.scheduled_job('cron', day_of_week='sun', hour=2, minute=0)
def weekly_job():
    print(f"[{datetime.now()}] ⏳ Weekly job started...")
    
    # Step 1: Fetch data
    try:
        print("📅 Auto Data Collection started.")
        refresher = subprocess.Popen(["python", "ServerSide/services/dataRefresh.py"], check=True )
        refresher.wait()
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")

    # Step 2: Train model
    try:
        print("📅 Auto trainer started.")
        subprocess.run(["python", "ServerSide/services/modelling.py"], check=True )
    except Exception as e:
        print(f"❌ Model retraining failed: {e}")

if __name__ == "__main__":
    scheduler.start()
