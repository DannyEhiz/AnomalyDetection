import schedule
import time
from advancedAlerting2 import (
    collectTelemetry,
    createOpenProblems,
    classifyBreachesForSlack,
    classifyBreachesForEmail,
    updateOpenProblems,
    sendToOpenProblemHandler,
    processAlert,
    format_and_send_alert_email,
    collectRegisteredAlert
)
import json
import shelve
import pandas as pd
import datetime
from datetime import timedelta, datetime
import sqlite3
import pyodbc
import warnings 
warnings.filterwarnings('ignore')
import schedule
import time
import os
import subprocess
from flask import Flask, jsonify
import threading
import logging
from logging.handlers import RotatingFileHandler
import slack
import ast

# Logger setup should run once per Streamlit rerun
logger = logging.getLogger('alert_logger')

# Avoid duplicate handlers on reruns
if logger.hasHandlers():
    logger.handlers.clear()
logger.setLevel(logging.DEBUG)
os.makedirs('logs', exist_ok=True)

# Formatter for logs
log_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(module)s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Rotating handler for general logs (INFO and above)
file_handler_info = RotatingFileHandler(
    'logs/app.log', maxBytes=5*1024*1024, backupCount=7
)
file_handler_info.setLevel(logging.INFO)
file_handler_info.setFormatter(log_formatter)

# Rotating handler for errors (ERROR and above)
file_handler_error = RotatingFileHandler(
    'logs/error.log', maxBytes=5*1024*1024, backupCount=10
)
file_handler_error.setLevel(logging.ERROR)
file_handler_error.setFormatter(log_formatter)

# Attach handlers to logger
logger.addHandler(file_handler_info)
logger.addHandler(file_handler_error)





def run_workflow():
    """
    Executes the entire workflow for monitoring and alerting.
    Handles exceptions at each step to ensure the workflow doesn't break.
    """
    createOpenProblems()
        
    newData = collectTelemetry()
    if newData is not None:
        print('Data upload detected -> checking threshold breach ...')
        updateOpenProblems(newData)

        slack_cpu, slack_disk, slack_mem = classifyBreachesForSlack(newData)
        email_cpu, email_disk, email_mem = classifyBreachesForEmail(newData)

        emailCPUbreach, emailDiskbreach, emailMembreach = sendToOpenProblemHandler(data_cpu=email_cpu, data_disk=email_disk, data_mem=email_mem)
        if len(emailCPUbreach) + len(emailDiskbreach) + len(emailMembreach) > 0:
            print('Threshold breach detected -> initiating notification system ...')   
                     
            # Process the remaining breaches into a structured format for email alerts
        outwardSendingConstructor = processAlert(cpubreach=emailCPUbreach, membreach=emailMembreach, diskbreach=emailDiskbreach)
        format_and_send_alert_email(outwardSendingConstructor)

        # if slack_cpu or slack_disk or slack_mem:
        #     slackCPUbreach, slackDiskbreach, slackMembreach = sendToOpenProblemHandler(slack_cpu, slack_disk, slack_mem)
    else:
        pass

while True:
    with sqlite3.connect('EdgeDB.db') as conn:
        conn.execute('PRAGMA journal_mode=WAL;')
        c = conn.cursor()
        d = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='openProblems';")
        openProblems_table_exists = c.fetchone()
        d.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alertUsers';")
        alertUsers_table_exists = d.fetchone()

        if not openProblems_table_exists and not alertUsers_table_exists:
            print(f"Required table for alert notification is not created yet. \nSleeping the notification system for 60 seconds...")
            time.sleep(60)
        else:
            alert_users = collectRegisteredAlert()

            if alert_users.empty:
                print(f"\nNo users registered for alerts. Sleeping notification system for 60 seconds...\n")
                time.sleep(60)
            else:
                try:
                    run_workflow()
                    sleep_duration = 30 # seconds
                    # print(f"Workflow finished. Sleeping for {sleep_duration} seconds...")
                    time.sleep(sleep_duration)
                except KeyboardInterrupt:
                    print("Stopping alerting workflow...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    print("An error occurred in the notification system. Sleeping for 60 seconds before retrying...")
                    time.sleep(60) 