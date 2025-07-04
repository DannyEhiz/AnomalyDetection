import sys 
import os
sys.path.append(os.path.abspath('.'))
import pandas as pd
from ServerSide.core.connection import fetchFromClientDB, saveToSQLite   
import sqlite3
from datetime import datetime, timedelta
import json
import time
import os
import gc
import logging
from ServerSide.core.tinyDBHandler import retrieveRecord, updateRecord, createRecord, tableIsExisting, removeItemFromRecord
from logging.handlers import RotatingFileHandler
from ServerSide.core.logger import logging_setup
logger = logging_setup(log_dir='logs/dataRefresh', 
                       general_log='refreshInfo.log', 
                       error_log='refreshError.log', 
                       loggerName='refreshLogger')


# Load configuration
with open('ServerSide/core/config.json') as config_file:
    configVar = json.load(config_file)
client_table_name1 = configVar['client_table_name1']
client_table_name2 = configVar['client_table_name2']


def create_refresh_logs_table():
    """Creates the RefreshLogs table if it doesn't exist."""
    with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS RefreshLogs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tableName TEXT,
                refresh_time TEXT,
                status TEXT,
                message TEXT
            )
        """)
        conn.commit()

def log_refresh(status: str, message: str):
    """Logs the data refresh event to the RefreshLogs table."""
    with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        c = conn.cursor()
        c.execute("""
            INSERT INTO RefreshLogs (tableName, refresh_time, status, message)
            VALUES (?, ?, ?, ?)
        """, (client_table_name1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, message))
        conn.commit()


def saveLastCollectionTime(data):
    with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
        c = conn.cursor()  
        try:
            if data is not None:
                data.to_sql('latestTime', conn, if_exists='replace', index=False) 
                gc.collect()
                del gc.garbage[:]
            else:
                print('Couldnt collect latest time from collected logs')
                pass
        except Exception as e:
            print(f"Error occurred while saving latest time to SQLite: {e}") 


def delete_old_rows():
    """
    Deletes rows older than 7 days from the SQLite database.
    """
    try:
        conn = sqlite3.connect('ServerSide/database/main.sqlite3')
        conn.execute('PRAGMA journal_mode=WAL')  # Enable Write-Ahead Logging
        cursor = conn.cursor()
        cursor.execute('SELECT MAX(LogTimestamp) FROM Infra_Utilization')
        maxTime = cursor.fetchone()
        sevenDaysAgo = (pd.to_datetime(maxTime) - timedelta(days=7)).strftime('%Y-%m-%s %H:%M:%S')
        cursor.execute("DELETE FROM Infra_Utilization WHERE LogTimestamp < ?", (sevenDaysAgo[0],))
        conn.commit()

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn: # type: ignore
            conn.close()


def refresh_data():
    """
    Refreshes telemetry data from the client database and updates the local SQLite storage.

    Workflow:
    - Initializes the refresh log table.
    - Fetches new data from the client database.
    - If the fetched data is empty, logs an error.
    - If data is collected:
        - Saves data to SQLite.
        - Checks if this is the first data collection:
            - If yes, creates tracking records in a TinyDB JSON file.
            - If no, saves the new batch to a temporary 'lastLog' table.
        - Ensures initial timestamp tracking is in place for anomaly detection.
        - Deletes old rows from the main storage to manage space.
    - Logs success or failure for each refresh attempt.
    
    This function is expected to run on a regular schedule to keep the local database updated for anomaly detection.
    """

    create_refresh_logs_table()
    print("Starting data refresh...")
    data = None
    try:
        data = fetchFromClientDB(client_table_name1, client_table_name2)
        if data.empty:
            log_refresh("Error", "Data collected is empty.")
            logger.error("Data table fetched from clientDB is empty. No new data is saved to tempDB.")
            print("Data table fetched from clientDB is empty. No new data is saved to tempDB.")
        else:
            saveToSQLite(data)
            # Check if the data is already existing at first. 
                # If its not, saves a tag into tinyDB json to signify data is now existence. Skips saving newly collected data to small miniDB
                # But if its existing, save the newly collected data to miniDB for ease of access 
            if not tableIsExisting('ServerSide/database/aux.json', 'dataPresent'):
                createRecord('ServerSide/database/aux.json', 'dataPresent', 'isDataPresent')
                updateRecord('ServerSide/database/aux.json', 'dataPresent', 'isDataPresent', True, append = False)
            else:
                with sqlite3.connect('ServerSide/database/mini.sqlite3') as conn:
                    c = conn.cursor()  
                    data.to_sql('lastLog', conn, if_exists='replace', index=False) 

            # Also if its first time collection, save the time stamp 
            # It will be updated in the run_modelling function 
            if not tableIsExisting('ServerSide/database/aux.json', 'latestLog'): 
                createRecord('ServerSide/database/aux.json', 'latestLog', 'logTime')

            delete_old_rows()
            del data
            print(f"Data fetched from client database at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
            log_refresh("Success", "Data refreshed and saved to SQLite.")
            logger.info(f"Data fetched from client database at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
            gc.collect()
            del gc.garbage[:]
    except Exception as e:
        log_refresh("Error", str(e))
        print(f"Error during data refresh: {e}")
        logger.error(f"Error during data refresh: {e}")
        

if __name__ == "__main__":
    refresh_data()
    from datetime import datetime, timedelta
    now = datetime.now()
    oneHourLater = now + timedelta(hours=1)
    print(f"Next refresh date is {oneHourLater.strftime('%A %Y-%m-%d at %H:%M')}")
    logger.info(f"Next refresh date is {oneHourLater.strftime('%A %Y-%m-%d at %H:%M')}")
    gc.collect()
    del gc.garbage[:]
