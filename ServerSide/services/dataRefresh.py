import pandas as pd
from ServerSide.core.connection import fetchFromClientDB, saveToSQLite   
import sqlite3
from datetime import datetime
import json
import time
import os
import gc
import logging
from logging.handlers import RotatingFileHandler
from core.logger import logging_setup
logger = logging_setup(log_dir='logs/dataRefresh', 
                       general_log='refreshInfo.log', 
                       error_log='refreshError.log', 
                       loggerName='refreshLogger')


# Load configuration
with open('core/config.json') as config_file:
    configVar = json.load(config_file)
client_table_name1 = configVar['client_table_name1']
client_table_name2 = configVar['client_table_name2']


def create_refresh_logs_table():
    """Creates the RefreshLogs table if it doesn't exist."""
    with sqlite3.connect('main.sqlite3') as conn:
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
    with sqlite3.connect('main.sqlite3') as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        c = conn.cursor()
        c.execute("""
            INSERT INTO RefreshLogs (tableName, refresh_time, status, message)
            VALUES (?, ?, ?, ?)
        """, (client_table_name1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, message))
        conn.commit()


def refresh_data():
    create_refresh_logs_table()
    print("Starting data refresh...")
    data = None
    try:
        data = fetchFromClientDB(client_table_name1, client_table_name2)
        if data.empty:
            log_refresh("Error", "Could not connect to client database.")
            logger.error("Data table fetched from clientDB is empty. No new data is saved to tempDB.")
            print("Data table fetched from clientDB is empty. No new data is saved to tempDB.")
        else:
            saveToSQLite(data)
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
    seven_days_later = now + timedelta(days=7)
    print(f"Next refresh date is {seven_days_later.strftime('%A %Y-%m-%d at %H:%M')}")
    logger.info(f"Next refresh date is {seven_days_later.strftime('%A %Y-%m-%d at %H:%M')}")
    gc.collect()
    del gc.garbage[:]
