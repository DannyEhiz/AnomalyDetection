from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from ServerSide.services.modelling import train_models 
import subprocess
from ServerSide.core.logger import logging_setup
logger = logging_setup(log_dir='logs/modelling', 
                       general_log='modellingInfo.log', 
                       error_log='modellingError.log', 
                       loggerName='modellingLogger')

scheduler = BlockingScheduler()


@scheduler.scheduled_job('interval', hours=1)
def hourlyJob():
    print(f"[{datetime.now()}] ⏳ hourly job started...")
    
    # Step 1: Fetch data
    try:
        print("📅 Auto Data Collection started.")
        refresher = subprocess.Popen(["python", "ServerSide/services/dataRefresh.py"], check=True )
        refresher.wait()
    except Exception as e:
        print(f"❌ Data refresh from client failed: {e}")
        logger.error(f"Data refresh from client failed: {e}")

    # Step 2: Train model
    try:
        print("📅 Auto trainer started.")
        subprocess.run(["python", "ServerSide/services/modelling.py"], check=True )
        print('✅✅✅General training complete')
        logger.info('General training complete')
    except Exception as e:
        print(f"❌ Model retraining failed: {e}")
        logger.error(f"Model retraining failed: {e}")

if __name__ == "__main__":
    scheduler.start()

