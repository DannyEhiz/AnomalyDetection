from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from ServerSide.core.connection import fetchFromClientDB
from ServerSide.services.modelling import train_models  # Adjust path if needed

scheduler = BlockingScheduler()

@scheduler.scheduled_job('cron', day_of_week='sun', hour=2, minute=0)
def weekly_job():
    print(f"[{datetime.now()}] ⏳ Weekly job started...")
    
    # Step 1: Fetch data
    try:
        fetchFromClientDB()
        print("✅ Data fetch successful.")
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return

    # Step 2: Train model
    try:
        train_models()
        print("✅ Model retraining successful.")
    except Exception as e:
        print(f"❌ Model retraining failed: {e}")

if __name__ == "__main__":
    print("📅 Scheduler started. Waiting for Sunday 2 AM...")
    scheduler.start()
