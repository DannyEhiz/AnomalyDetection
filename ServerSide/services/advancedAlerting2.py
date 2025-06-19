import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import google.generativeai as genai
from sysAlert import infraModelInstructionSlack, infraModelInstructionHTML
import gc
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
from waitress import serve
import slack
import html2text
import ast


# Config Variables 
with open('config.json') as config_file:
    configVar = json.load(config_file)

ClientServer = configVar['client_server']
ClientDB = configVar['client_db']
ClientDBUserName = configVar['client_db_username']
ClientDBPass = configVar['client_db_password']
Client_table_name1 = configVar['client_table_name1']
Client_table_name2 = configVar['client_table_name2']
driver = configVar['driver']
ClientDBPort = configVar['client_db_port']
slack_token = configVar['slack_token']
client = slack.WebClient(token=slack_token)


# === Logging Setup ===
LOG_DIR = 'logs/alerts'
GENERAL_LOG = os.path.join(LOG_DIR, 'dashboardAlerting.log')
ERROR_LOG = os.path.join(LOG_DIR, 'dashboardAlerting.log')

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)
# === Utility to clear logs once per 3day ===
def clear_log_daily(log_file_path):
    """Clear the log file if it's older than three days."""
    if os.path.exists(log_file_path):
        last_modified = datetime.fromtimestamp(os.path.getmtime(log_file_path))
        now = datetime.now()
        # Check if the log file is older than three days
        if last_modified < now - timedelta(days=7):
            with open(log_file_path, 'w'):
                pass  # This truncates the 

# Clear logs if they're from an older day
clear_log_daily(ERROR_LOG)
clear_log_daily(GENERAL_LOG)

# === Logger Configuration ===
logger = logging.getLogger('dashboardAlert_logger')

# Clear existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.DEBUG)

# Formatter for logs
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(module)s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Handler for INFO and above
info_handler = RotatingFileHandler(GENERAL_LOG, maxBytes=5*1024*1024, backupCount=1)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

# Handler for ERROR and above
error_handler = RotatingFileHandler(ERROR_LOG, maxBytes=5*1024*1024, backupCount=1)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Attach handlers to logger
logger.addHandler(info_handler)
logger.addHandler(error_handler)



def connectClientDB(server: str, database: str, username: str, password: str) -> str:
    connection_string = (
        f"Driver={driver};"
        f"Server=tcp:{ClientServer},{ClientDBPort};"
        f"Database={ClientDB};"
        f"Uid={ClientDBUserName};"
        f"Pwd={ClientDBPass};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=yes;"
        f"Connection Timeout=30;"
    )
    try:
        conn = pyodbc.connect(connection_string)
        return conn
    except Exception as e:
        print("Couldnt connect to database: ",e)
        return None

def createOpenProblems():
    """ Creates the openProblems table in the local SQLite DB if it doesn't exist. """
    try:
        with sqlite3.connect('EdgeDB.db') as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS openProblems (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_username TEXT NOT NULL,
                    server TEXT NOT NULL,
                    drive TEXT NOT NULL,
                    metric TEXT NOT NULL, -- 'CPU', 'MEM', 'DISK'
                    breached_value REAL,
                    threshold_value REAL,
                    first_breach_date TEXT NOT NULL,
                    time_active TEXT NOT NULL,
                    status TEXT DEFAULT 'OPEN')""")
            conn.commit()
    except sqlite3.Error as e:
        print(e)


def penultimateLog():
    with sqlite3.connect('EdgeDB.db') as conn:
        query = f"""select last_update_time from latestLogTime
                    order by last_update_time desc limit 1;
                """
        cursor = conn.cursor()
        cursor.execute(query)
        last_update_time = cursor.fetchone()[0]
    return last_update_time

def collectLastRefreshTime():
    with sqlite3.connect('EdgeDB.db') as conn:
        query = f"""select max(LogTimestamp) from Infra_Utilization
                """
        cursor = conn.cursor()
        cursor.execute(query) 
        lastTime = cursor.fetchone()[0]
    return lastTime
def checkIfDataIsUpdated():
    data_is_updated = False
    with shelve.open('alertDB.db') as db:
        if 'lastLog' not in db:
            db['lastLog'] = penultimateLog()
            # 'lastLog' will not be there, so data_is_updated comes out False. Solves problem of cold start
        else:
            penultimateTime = db['lastLog']
            currentTime = collectLastRefreshTime()
            if currentTime != penultimateTime:
                data_is_updated = True
                #! Dont update the penultimateLog() yet coz we need the time for fetching recently added data
    return data_is_updated


def collectRegisteredAlert():
    with sqlite3.connect('EdgeDB.db') as conn:
        data = pd.read_sql_query('select * from alertUsers where Active = 1', conn)
        return data
def openProbs():
    with sqlite3.connect('EdgeDB.db') as conn:
        return pd.read_sql_query('select * from openProblems', conn)
def checkOpenProblem():
    with sqlite3.connect('EdgeDB.db') as conn:
        data = pd.read_sql_query('select * from openProblems', conn)
        return data
   

def collectTelemetry(): # Collects newly added data
    if checkIfDataIsUpdated():
        with shelve.open('alertDB.db') as db:
            latestLog = db['lastLog']
        currentTime = collectLastRefreshTime()
        try:
            with sqlite3.connect('EdgeDb.db') as conn:
                query=f"SELECT LogTimestamp, Hostname, IPAddress, CPUUsage, MemoryUsage, TotalFreeDiskGB, ManagementZone,  DriveLetter FROM Infra_Utilization WHERE LogTimestamp > '{latestLog}'"
                data = pd.read_sql_query(query, conn)
                return data
        except Exception as e:
            print(f'Couldnt collect data from sqlite3: At CollectTelemetry: {e}')
        finally:
            with shelve.open('alertDB.db') as db:
                db['lastLog'] = currentTime  # Update the penultimateLog() to the current time
    else:
        return None
    
def checkBreaches(latest):
    """
    This function identifies resource utilization breaches (CPU, Memory, Disk) by comparing the latest telemetry data 
    with registered alert thresholds. It merges telemetry data with alert configurations, checks for threshold violations, 
    and returns the rows where any breach has occurred.

    Steps:
    1. **Collect Latest Telemetry Data**:
       - Calls `collectTelemetry()` to retrieve the most recent resource utilization data from the database.
    
    2. **Retrieve Registered Alerts**:
       - Calls `collectRegisteredAlert()` to fetch the alert configurations for active users from the database.

    3. **Merge Telemetry Data with Alert Configurations**:
       - Merges the telemetry data (`latest`) with the alert configurations (`alerts`) on the `ManagementZone` column 
         from telemetry and `MgtZone` column from alerts. This ensures that telemetry data is matched with the 
         corresponding alert thresholds.

    4. **Remove Duplicate Rows**:
       - Removes duplicate rows from the merged DataFrame to ensure unique entries.

    5. **Check for Breaches**:
       - Compares the telemetry data with the alert thresholds to identify breaches:
         - **CPU Breach**: Checks if `CPUUsage` exceeds `CPU_thresh`.
         - **Memory Breach**: Checks if `MemoryUsage` exceeds `MEM_thresh`.
         - **Disk Breach**: Checks if `TotalFreeDiskGB` is less than `DISK_thresh`.

    6. **Filter Breached Rows**:
       - Filters the rows where any of the above breaches (CPU, Memory, Disk) have occurred.

    7. **Return Breaches**:
       - Returns the filtered DataFrame containing only the rows where breaches have been detected.

    Returns:
        pd.DataFrame: A DataFrame containing rows where resource utilization breaches have occurred.

    Example Output:
        The returned DataFrame will include columns such as `CPU_Breach`, `MEM_Breach`, and `DISK_Breach` 
        indicating whether a breach occurred for each metric.

    Dependencies:
        - `collectTelemetry()`: Fetches the latest telemetry data.
        - `collectRegisteredAlert()`: Fetches the registered alert configurations.
    """
    alerts = collectRegisteredAlert()
    if latest is not None and alerts is not None :
        mergedf = pd.merge(latest, alerts, how='inner', left_on='ManagementZone', right_on='MgtZone')
        mergedf.drop_duplicates(inplace=True)
        # Check if metrics exceed thresholds
        mergedf['CPU_Breach'] = mergedf['CPUUsage'].astype(int) > mergedf['CPU_thresh'].astype(int)
        mergedf['MEM_Breach']= mergedf['MemoryUsage'].astype(int) > mergedf['MEM_thresh'].astype(int)
        mergedf['DISK_Breach'] = mergedf['TotalFreeDiskGB'].astype(int) < mergedf['DISK_thresh'].astype(int)
        # Filter rows where any threshold is breached
        breaches = mergedf[(mergedf['CPU_Breach']) | (mergedf['MEM_Breach']) | (mergedf['DISK_Breach'])]
        return breaches
    else:
        return None

def classifyBreachesForSlack(latest):
    """
    This function filters and classifies resource utilization breaches (CPU, Memory, Disk) specifically for Slack notifications. 
    It processes the breached data returned by `checkBreaches()` and organizes it into separate DataFrames for each metric type.

    Steps:
    1. **Retrieve Breached Data**:
       - Calls `checkBreaches()` to get the DataFrame containing all resource utilization breaches.

    2. **Filter for Slack Notifications**:
       - Filters the breached data to include only rows where the `AlertType` is set to 'slack'.

    3. **Classify Breaches by Metric Type**:
       - Separates the filtered data into three categories:
         - **CPU Breach**: Rows where `CPU_Breach` is `True`.
         - **Memory Breach**: Rows where `MEM_Breach` is `True`.
         - **Disk Breach**: Rows where `DISK_Breach` is `True`.

    4. **Exclude Unnecessary Columns**:
       - For each breach type, excludes columns that are not relevant for Slack notifications, such as thresholds for other metrics and metadata.

    5. **Return Classified Breaches**:
       - Returns three DataFrames: one for CPU breaches, one for Memory breaches, and one for Disk breaches.

    Returns:
        tuple: A tuple containing three DataFrames:
            - `cpubreach`: DataFrame for CPU breaches.
            - `diskbreach`: DataFrame for Memory breaches.
            - `membreach`: DataFrame for Disk breaches.

    Example Output:
        Each returned DataFrame will contain only the relevant columns for the specific breach type and rows where the breach occurred.

    Dependencies:
        - `checkBreaches()`: Fetches the breached data.
    """
    data = checkBreaches(latest)
    if data is not None or not data.empty:
        data = data[data.AlertType == 'slack']
        cpubreach = data[data['CPU_Breach']][[i for i in data if i not in ['MemoryUsage', 'TotalFreeDiskGB', 'Active', 'MgtZone', 
                                                                                        'MEM_thresh', 'DISK_thresh', 'dateCreated', 
                                                                                    'MEM_Breach', 'DISK_Breach', 'CPU_Breach' ]]]
        diskbreach = data[data['MEM_Breach']][[i for i in data if i not in ['MemoryUsage', 'CPUUsage', 'Active', 'MgtZone', 
                                                                                    'CPU_thresh', 'MEM_thresh',  'dateCreated', 
                                                                                    'MEM_Breach', 'DISK_Breach' , 'CPU_Breach']]]
        membreach = data[data['DISK_Breach']][[i for i in data if i not in ['CPUUsage', 'TotalFreeDiskGB', 'Active', 'MgtZone', 
                                                                                    'CPU_thresh',  'DISK_thresh', 'dateCreated', 
                                                                                    'MEM_Breach', 'DISK_Breach', 'CPU_Breach' ]]]
        return cpubreach, diskbreach, membreach
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        

def classifyBreachesForEmail(latest):
    """
    This function filters and classifies resource utilization breaches (CPU, Memory, Disk) specifically for Slack notifications. 
    It processes the breached data returned by `checkBreaches()` and organizes it into separate DataFrames for each metric type.

    Steps:
    1. **Retrieve Breached Data**:
       - Calls `checkBreaches()` to get the DataFrame containing all resource utilization breaches.

    2. **Filter for Slack Notifications**:
       - Filters the breached data to include only rows where the `AlertType` is set to 'slack'.

    3. **Classify Breaches by Metric Type**:
       - Separates the filtered data into three categories:
         - **CPU Breach**: Rows where `CPU_Breach` is `True`.
         - **Memory Breach**: Rows where `MEM_Breach` is `True`.
         - **Disk Breach**: Rows where `DISK_Breach` is `True`.

    4. **Exclude Unnecessary Columns**:
       - For each breach type, excludes columns that are not relevant for Slack notifications, such as thresholds for other metrics and metadata.

    5. **Return Classified Breaches**:
       - Returns three DataFrames: one for CPU breaches, one for Memory breaches, and one for Disk breaches.

    Returns:
        tuple: A tuple containing three DataFrames:
            - `cpubreach`: DataFrame for CPU breaches.
            - `diskbreach`: DataFrame for Memory breaches.
            - `membreach`: DataFrame for Disk breaches.

    Example Output:
        Each returned DataFrame will contain only the relevant columns for the specific breach type and rows where the breach occurred.

    Dependencies:
        - `checkBreaches()`: Fetches the breached data.
    """
    data = checkBreaches(latest)
    if data is not None or not data.empty:
        data = data[data.AlertType == 'email']
        cpubreach = data[data['CPU_Breach']][[i for i in data if i not in ['MemoryUsage', 'TotalFreeDiskGB', 'Active', 'MgtZone', 
                                                                                        'MEM_thresh', 'DISK_thresh', 'dateCreated', 
                                                                                    'MEM_Breach', 'DISK_Breach', 'CPU_Breach' ]]]
        diskbreach = data[data['DISK_Breach']][[i for i in data if i not in ['MemoryUsage', 'CPUUsage', 'Active', 'MgtZone', 
                                                                                    'CPU_thresh', 'MEM_thresh',  'dateCreated', 
                                                                                    'MEM_Breach', 'DISK_Breach' , 'CPU_Breach']]]
        membreach = data[data['MEM_Breach']][[i for i in data if i not in ['CPUUsage', 'TotalFreeDiskGB', 'Active', 'MgtZone', 
                                                                                    'CPU_thresh',  'DISK_thresh', 'dateCreated', 
                                                                                    'MEM_Breach', 'DISK_Breach', 'CPU_Breach' ]]]
        return cpubreach, diskbreach, membreach
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        

# def updateOpenProblems(newData):
#     """
#     This function updates the status of open problems in the `openProblems` table by checking if the breached metrics 
#     (CPU, Disk, Memory) have returned to normal levels. If a metric is now below its threshold, the corresponding 
#     problem's status is updated to 'CLOSED'.

#     Steps:
#     1. **Define Helper Function**:
#        - `updateRowStatus(id)`: Updates the status of a specific row in the `openProblems` table to 'CLOSED'.

#     2. **Fetch Latest Telemetry Data**:
#        - Calls `collectTelemetry()` to retrieve the most recent resource utilization data from the database.

#     3. **Fetch Open Problems**:
#        - Calls `checkOpenProblem()` to retrieve the current open problems from the `openProblems` table.

#     4. **Define Metric Mapping**:
#        - Maps metric types (`cpu`, `disk`, `memory`) to their respective column names in the telemetry data.

#     5. **Iterate Through Metrics**:
#        - For each metric type, iterates through the telemetry data and open problems to check if the metric has returned 
#          to normal levels.

#     6. **Check and Update Status**:
#        - Compares the telemetry data with the threshold values in the `openProblems` table. If the metric is below the 
#          threshold, the status of the corresponding problem is updated to 'CLOSED' using `updateRowStatus()`.

#     Dependencies:
#         - `collectTelemetry()`: Fetches the latest telemetry data.
#         - `checkOpenProblem()`: Fetches the current open problems from the database.

#     Example Scenario:
#         If a server's CPU usage was previously above the threshold and is now below the threshold, the corresponding 
#         problem's status in the `openProblems` table will be updated to 'CLOSED'.
#     """
#     def updateRowStatus(id):
#         with sqlite3.connect('EdgeDB.db') as conn:
#             cur = conn.cursor()
#             cur.execute('UPDATE openProblems SET status = ? WHERE id = ?', ('CLOSED', id))
#             conn.commit()
#     # Fetch the latest telemetry data and open problems
#     try:
#         probs = checkOpenProblem()
#         # Define the mapping of metrics to their respective columns
#         metricsType = {'cpu': 'CPUUsage', 
#                     'disk': 'TotalFreeDiskGB', 
#                     'memory': 'MemoryUsage'}
#         if not newData.empty:
#             newdata_set = set((row['Hostname'], row['DriveLetter']) for _, row in newData.iterrows())
#         probs_isEmpty = probs.empty
#         newData_isEmpty = newData.empty

#         for key, value in metricsType.items():
#             if not probs_isEmpty and not newData_isEmpty:
#                 for _, row in probs[(probs.metric==key) and (probs.status == 'OPEN')].iterrows():
#                     if (row['server'], row['drive']) in newdata_set:
#                         if row['threshold'] <

#         if newData is not None and not newData.empty and not probs.empty:
#             # Iterate through each metric type
#             for key, value in metricsType.items():
#                 for _, row in newData.iterrows():
#                     for _, row2 in probs.iterrows():
#                         # Check if the hostname, drive, and metric match
#                         if row['Hostname'] == row2['server'] and row['DriveLetter'] == row2['drive'] and row2['metric'] == key and row2['status'] == 'OPEN':
#                             # If the metric is now below the threshold, update the status to CLOSED
#                             if row[value] < row2['threshold_value']:
#                                 updateRowStatus(row2['id'])
#     except Exception as e:
#         print(f'An error occurred at updateOpenProblems\nProblem not updated to CLOSE on openProblem table\n', e)

def updateOpenProblems(newData):
    """
    This function updates the status of open problems in the `openProblems` table by checking if the breached metrics 
    (CPU, Disk, Memory) have returned to normal levels. If a metric is now below its threshold, the corresponding 
    problem's status is updated to 'CLOSED'.

    Steps:
    1. **Define Helper Function**:
       - `updateRowStatus(id)`: Updates the status of a specific row in the `openProblems` table to 'CLOSED'.

    2. **Fetch Latest Telemetry Data**:
       - Calls `collectTelemetry()` to retrieve the most recent resource utilization data from the database.

    3. **Fetch Open Problems**:
       - Calls `checkOpenProblem()` to retrieve the current open problems from the `openProblems` table.

    4. **Define Metric Mapping**:
       - Maps metric types (`cpu`, `disk`, `memory`) to their respective column names in the telemetry data.

    5. **Iterate Through Metrics**:
       - For each metric type, iterates through the telemetry data and open problems to check if the metric has returned 
         to normal levels.

    6. **Check and Update Status**:
       - Compares the telemetry data with the threshold values in the `openProblems` table. If the metric is below the 
         threshold, the status of the corresponding problem is updated to 'CLOSED' using `updateRowStatus()`.

    Dependencies:
        - `collectTelemetry()`: Fetches the latest telemetry data.
        - `checkOpenProblem()`: Fetches the current open problems from the database.

    Example Scenario:
        If a server's CPU usage was previously above the threshold and is now below the threshold, the corresponding 
        problem's status in the `openProblems` table will be updated to 'CLOSED'.
    """
    def updateRowStatus(id):
        try:
            with sqlite3.connect('EdgeDB.db') as conn:
                cur = conn.cursor()
                cur.execute('UPDATE openProblems SET status = ? WHERE id = ?', ('CLOSED', id))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating row status for id {id}: {e}", exc_info=True)
            print(f'Error updating row status for id {id}: {e}')

    # Check if there are no duplicate open problems ( duplicate open problems are items that have matching alert_username, server, drive, and metric then keep only the latest)
    openProblems = checkOpenProblem()
    openProblems_open = openProblems[openProblems['status'] == 'OPEN'].sort_values(by = 'first_breach_date', ascending=True)
    openProblems_close = openProblems[openProblems['status'] == 'CLOSED'].sort_values(by = 'first_breach_date', ascending=True)
    
    if openProblems_open.duplicated(subset=['alert_username', 'server', 'drive', 'metric']).any():
        openProblems_open = openProblems_open.drop_duplicates(subset=['alert_username', 'server', 'drive', 'metric'], keep='last')

        openProblems = pd.concat([openProblems_open, openProblems_close], axis = 0)
        
        if 'id' in openProblems.columns:
            openProblems = openProblems.drop(columns=['id'])

        try:
            with sqlite3.connect('EdgeDB.db') as conn:
                cursor = conn.cursor()
                cursor.execute("DROP TABLE IF EXISTS openProblems")
                cursor.execute("""
                    CREATE TABLE openProblems (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_username TEXT NOT NULL,
                        server TEXT NOT NULL,
                        drive TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        breached_value REAL,
                        threshold_value REAL,
                        first_breach_date TEXT NOT NULL,
                        time_active TEXT NOT NULL,
                        status TEXT DEFAULT 'OPEN'
                    )
                """)

                openProblems.to_sql('openProblems', conn, if_exists='append', index=False)

        except Exception as e:
            print('Encountered an error while trying to rebuild openProblems table:', e)
        # try:
        #     with sqlite3.connect('EdgeDB.db') as conn:
        #         openProblems.to_sql('openProblems', conn, if_exists='replace', index=False)
        # except:
        #     print('Encountered an error while trying to remove duplicate problems from the problems table.. Check updateOpenProblem() function', e)
        #     pass

    openProblems, openProblems_close, openProblems_open = None, None, None
    del openProblems, openProblems_close, openProblems_open
    gc.collect()

    try:
        probs = checkOpenProblem()
        metricsType = {'cpu': 'CPUUsage',
                       'disk': 'TotalFreeDiskGB', # Disk check is based on free space
                       'memory': 'MemoryUsage'}
        if newData is not None and not newData.empty and not probs.empty:
            # Iterate through open problems instead of new data for efficiency
            for _, prob_row in probs[probs['status'] == 'OPEN'].iterrows():
                server = prob_row['server']
                drive = prob_row['drive']
                metric = prob_row['metric'].lower() # Ensure lowercase
                threshold = prob_row['threshold_value']
                problem_id = prob_row['id']

                # Find the latest entry in newData for this server/drive
                latest_server_data = newData[(newData['Hostname'] == server) & (newData['DriveLetter'] == drive)].sort_values('LogTimestamp', ascending=False).iloc[:1]

                if not latest_server_data.empty:
                    latest_row = latest_server_data.iloc[0]
                    metric_col = metricsType.get(metric) #retrieves the columnName (as entered in metricType) for the current metric type been considered

                    if metric_col:
                        try:
                            current_value = pd.to_numeric(latest_row[metric_col], errors='coerce')
                            if pd.isna(current_value):
                                logger.warning(f"Could not convert {metric_col} value '{latest_row[metric_col]}' to numeric for problem ID {problem_id}.")
                                continue

                            # Check for resolution based on metric type
                            is_resolved = False
                            if metric == 'disk': # Disk is resolved if free space is >= threshold
                                if current_value >= threshold:
                                    is_resolved = True
                            else: # CPU and Memory are resolved if value <= threshold
                                if current_value <= threshold:
                                    is_resolved = True

                            if is_resolved:
                                updateRowStatus(problem_id)
                                logger.info(f"Problem ID {problem_id} ({server}/{drive}/{metric}) resolved and marked as CLOSED.")
                                print(f"Problem ID {problem_id} ({server}/{drive}/{metric}) resolved and marked as CLOSED.")

                        except KeyError:
                            logger.error(f"Metric column '{metric_col}' not found in newData for problem ID {problem_id}.")
                            print(f"Metric column '{metric_col}' not found in newData for problem ID {problem_id}.")
                        except Exception as e:
                            logger.error(f"Error checking resolution for problem ID {problem_id}: {e}", exc_info=True)
                            print(f"Error checking resolution for problem ID {problem_id}: {e}")

    except Exception as e:
        logger.error(f'An unexpected error occurred in updateOpenProblems: {e}', exc_info=True)
        print(f'An unexpected error occurred in updateOpenProblems: {e}')

def sendToOpenproblems(breachData, metricType, metricColName, threshColName):
    """
    This function processes breached data and updates the `openProblems` table in the database. It checks if the breached 
    metrics (CPU, Disk, Memory) are already recorded as open problems. If a breach is already recorded, it skips adding 
    it again. Otherwise, it inserts the new breach into the `openProblems` table.

    Steps:
    1. **Parse Server List**:
       - Converts the `Server_List` column from a string to a Python list using `ast.literal_eval`.

    2. **Explode Server List**:
       - Expands the `Server_List` column so that each server in the list becomes a separate row.

    3. **Fetch Open Problems**:
       - Calls `checkOpenProblem()` to retrieve the current open problems from the `openProblems` table.

    4. **Check for Existing Problems**:
       - Iterates through the breached data and the open problems to check if the breach is already recorded. 
         If the breach is already recorded and its status is 'OPEN', it is skipped.

    5. **Insert New Problems**:
       - For breaches not already recorded, calculates the time difference between the current time and the log timestamp 
         of the breach. Inserts the new breach into the `openProblems` table with the calculated time difference.

    6. **Remove Processed Rows**:
       - Removes rows from the breached data that are already recorded in the `openProblems` table.

    7. **Return Remaining Breaches**:
       - Returns the remaining breached data that was not already recorded in the `openProblems` table.

    Args:
        breachData (pd.DataFrame): The DataFrame containing breached data.
        metricType (str): The type of metric (e.g., 'cpu', 'disk', 'memory').
        metricColName (str): The column name for the metric value in the breached data.
        threshColName (str): The column name for the threshold value in the breached data.

    Returns:
        pd.DataFrame: A DataFrame containing the remaining breached data that was not already recorded in the `openProblems` table.

    Dependencies:
        - `checkOpenProblem()`: Fetches the current open problems from the database.
        - `sqlite3`: Used for database operations.
        - `ast.literal_eval`: Safely evaluates strings containing Python literals.

    Example Usage:
        remainingBreaches = sendToOpenproblems(breachData, 'cpu', 'CPUUsage', 'CPU_thresh')

    Example Scenario:
        If a server's CPU usage exceeds the threshold and is not already recorded in the `openProblems` table, 
        this function will add it to the table and return the remaining breaches.
    """
    # Checks if in problems, if in problems, deletes the row, if not there, sends to problems 
    try:
        if breachData is not None or not breachData.empty:
            breachData['Server_List'] = breachData['Server_List'].apply(lambda x:  ast.literal_eval(x))
            breachData = breachData.explode('Server_List')
            breachData = breachData[breachData.Hostname == breachData.Server_List]

            indexToRemove = []
            
            openProbs = checkOpenProblem()
            open_problems_set = set(
                    (row['server'], row['metric'], row['drive'])
                    for _, row in openProbs[(openProbs.status == 'OPEN') & (openProbs.metric == metricType)].iterrows()
                )
            for index, row in breachData.iterrows():
                if (row['Server_List'], metricType, row['DriveLetter']) in open_problems_set:
                    indexToRemove.append(index)

                else:
                    log_time = datetime.strptime(row['LogTimestamp'], "%Y-%m-%d %H:%M:%S")
                    now = datetime.now()
                    time_difference = abs(now - log_time)
                    hours, remainder = divmod(time_difference.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    times = f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
                    with sqlite3.connect('EdgeDB.db') as conn:
                        c = conn.cursor()
                        c.execute("INSERT INTO openProblems (alert_username, server, drive, metric, breached_value, threshold_value, first_breach_date, time_active, status) VALUES (?, ?, ?, ?,?, ?, ?, ?, ?)",
                        (row['Username'], row['Server_List'], row['DriveLetter'], metricType, row[metricColName], row[threshColName], row['LogTimestamp'], times, 'OPEN'))   
            
            
            # for index2, row2 in breachData.iterrows():
            #     for index, row in openProbs[(openProbs.status == 'OPEN') & (openProbs.metric == metricType)].iterrows():
            #         if row2['Server_List']==row['server'] and row2['DriveLetter']==row['drive']:
            #             indexToRemove.append(index2)
            #         else: 
            #             log_time = datetime.strptime(row2['LogTimestamp'], "%Y-%m-%d %H:%M:%S")
            #             now = datetime.now()
            #             time_difference = abs(now - log_time)
            #             hours, remainder = divmod(time_difference.total_seconds(), 3600)
            #             minutes, seconds = divmod(remainder, 60)
            #             times = f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
            #             with sqlite3.connect('EdgeDB.db') as conn:
            #                 c = conn.cursor()
            #                 c.execute("INSERT INTO openProblems (alert_username, server, drive, metric, breached_value, threshold_value, first_breach_date, time_active, status) VALUES (?, ?, ?, ?,?, ?, ?, ?, ?)",
            #                         (row2['Username'], row2['Server_List'], row2['DriveLetter'], metricType, row2[metricColName], row2[threshColName], row2['LogTimestamp'], times, 'OPEN'))
            breachData = breachData.drop(index=indexToRemove)
            return breachData # returns new breach data that are not in the open problems
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f'An error in sendToOpenproblems function: \nCouldnt send problems to openProblem table\n', e)
        return pd.DataFrame()


def sendToOpenProblemHandler(data_cpu, data_disk, data_mem):
    out1 = sendToOpenproblems(data_cpu, 'cpu', 'CPUUsage', 'CPU_thresh' )
    out2 = sendToOpenproblems(data_disk, 'disk', 'TotalFreeDiskGB', 'DISK_thresh')
    out3 = sendToOpenproblems(data_mem, 'memory', 'MemoryUsage', 'MEM_thresh')
    return out1, out2, out3


# def processAlert(cpubreach, diskbreach, membreach):
#     """
#     This function processes breached metrics (CPU, Disk, Memory) and organizes them into a structured format for sending alerts. 
#     It groups the data by users and prepares it for email or Slack notifications.

#     Steps:
#     1. **Initialize Metrics**:
#        - Creates a dictionary `metrics` containing the breached data for CPU, Disk, and Memory.

#     2. **Iterate Through Metrics**:
#        - Loops through each metric type (CPU, Disk, Memory) and its corresponding breached data.

#     3. **Group Data by User**:
#        - For each metric type, groups the breached data by user and extracts the list of servers associated with each user.

#     4. **Determine Metric Column**:
#        - Identifies the column name for the metric value based on the metric type (e.g., `CPUUsage` for CPU).

#     5. **Structure Alert Details**:
#        - For each user, iterates through their breached data and structures the alert details into a dictionary format.

#     6. **Store Alerts**:
#        - Stores the structured alert details in a nested dictionary `tobesent`, organized by metric type and user.

#     7. **Return Structured Alerts**:
#        - Returns the `tobesent` dictionary containing all the structured alert data.

#     Args:
#         cpubreach (pd.DataFrame): DataFrame containing CPU breaches.
#         diskbreach (pd.DataFrame): DataFrame containing Disk breaches.
#         membreach (pd.DataFrame): DataFrame containing Memory breaches.

#     Returns:
#         dict: A nested dictionary containing structured alert data, grouped by metric type and user.

#     Example Output:
#         {
#             'CPU': {
#                 'user1': [
#                     {
#                         "Hostname": "server1",
#                         "IPAddress": ["192.168.1.1"],
#                         "Metric": "CPU",
#                         "MetricValue": 95,
#                         "ManagementZone": "Zone1",
#                         "DriveLetter": "C",
#                         "LogTimestamp": "2025-04-17 10:00:00",
#                         "Emails": ["user1@example.com"]
#                     },
#                     ...
#                 ],
#                 ...
#             },
#             'Disk': { ... },
#             'Memory': { ... }
#         }

#     Dependencies:
#         - `ast.literal_eval`: Safely evaluates strings containing Python literals.
#         - `pandas`: Used for DataFrame operations such as filtering and iteration.

#     Example Usage:
#         structured_alerts = processAlert(cpubreach, diskbreach, membreach)
#         print(structured_alerts)
#     """
#     # try:
#     metrics = {'CPU': cpubreach, 'Disk': diskbreach, 'Memory': membreach}
#     tobesent = {}
#     # Loop through each metric type (CPU, Disk, Memory) and its corresponding breached data.
#     for key, metric in metrics.items():
#         if not metric.empty:
#             user_server = {}  # Temporary dictionary to map users to their servers.
#             tobesent[key] = {}  # Create a nested dictionary for each metric type in `tobesent`.
        
#             users = metric.Username.unique().tolist()  # Get the list of unique users associated with the current metric type.

#             for i in users: # Loop through each user to process their data.
#                 # Extract the list of servers associated with the user and convert it from a string to a Python list.
#                 def literal(x):
#                     if isinstance(x, str):
#                         return ast.literal_eval(x)
#                     elif isinstance(x, list):
#                         return x 
        
#                 user_server[i] = ast.literal_eval(metric[metric.Username == i].iloc[0]['Server_List']) if isinstance(metric[metric.Username == i].iloc[0]['Server_List'], str) else metric[metric.Username == i].iloc[0]['Server_List']
                
#                 # Filter the data for the current user.
#                 checkData = metric[metric.Username == i]
                
#                 # Initialize an empty list to store the user's alerts for the current metric type.
#                 tobesent[key][i] = []
                
#                 # Determine the column name for the metric value based on the metric type.
#                 if key == 'CPU':
#                     metricName = 'CPUUsage'
#                 elif key == 'Disk':
#                     metricName = 'TotalFreeDiskGB'
#                 elif key == 'Memory':
#                     metricName = 'MemoryUsage'

#                 # Loop through each row of the user's data to extract and structure the alert details.
#                 for _, row in checkData.iterrows():
#                     # Append a dictionary containing the alert details to the user's list in `tobesent`.
#                     tobesent[key][i].append({
#                         "Hostname": row.Hostname,  # The server's hostname.
#                         "IPAddress": ast.literal_eval(row.IPAddress_x),  # The server's IP address (converted to a list).
#                         "Metric": key,  # The type of metric (CPU, Disk, or Memory).
#                         "MetricValue": row[metricName],  # The value of the breached metric.
#                         "ManagementZone": row.ManagementZone,  # The management zone of the server.
#                         "DriveLetter": row.DriveLetter,  # The drive letter (for Disk metrics).
#                         "LogTimestamp": row.LogTimestamp,  # The timestamp of the log entry.
#                         "Emails": ast.literal_eval(row.Emails)  # The list of email addresses to notify.
#                     })
#     return tobesent
    # except Exception as e:
    #     print(f"An error occurred at processAlert function\nCouldnt processAlert for sending\n: {e}")

def processAlert(cpubreach, membreach, diskbreach): # Corrected order
    """
    This function processes breached metrics (CPU, Disk, Memory) and organizes them into a structured format for sending alerts. 
    It groups the data by users and prepares it for email or Slack notifications.

    Steps:
    1. **Initialize Metrics**:
       - Creates a dictionary `metrics` containing the breached data for CPU, Disk, and Memory.

    2. **Iterate Through Metrics**:
       - Loops through each metric type (CPU, Disk, Memory) and its corresponding breached data.

    3. **Group Data by User**:
       - For each metric type, groups the breached data by user and extracts the list of servers associated with each user.

    4. **Determine Metric Column**:
       - Identifies the column name for the metric value based on the metric type (e.g., `CPUUsage` for CPU).

    5. **Structure Alert Details**:
       - For each user, iterates through their breached data and structures the alert details into a dictionary format.

    6. **Store Alerts**:
       - Stores the structured alert details in a nested dictionary `tobesent`, organized by metric type and user.

    7. **Return Structured Alerts**:
       - Returns the `tobesent` dictionary containing all the structured alert data.

    Args:
        cpubreach (pd.DataFrame): DataFrame containing CPU breaches.
        diskbreach (pd.DataFrame): DataFrame containing Disk breaches.
        membreach (pd.DataFrame): DataFrame containing Memory breaches.

    Returns:
        dict: A nested dictionary containing structured alert data, grouped by metric type and user.

    Example Output:
        {
            'CPU': {
                'user1': [
                    {
                        "Hostname": "server1",
                        "IPAddress": ["192.168.1.1"],
                        "Metric": "CPU",
                        "MetricValue": 95,
                        "ManagementZone": "Zone1",
                        "DriveLetter": "C",
                        "LogTimestamp": "2025-04-17 10:00:00",
                        "Emails": ["user1@example.com"]
                    },
                    ...
                ],
                ...
            },
            'Disk': { ... },
            'Memory': { ... }
        }

    Dependencies:
        - `ast.literal_eval`: Safely evaluates strings containing Python literals.
        - `pandas`: Used for DataFrame operations such as filtering and iteration.

    Example Usage:
        structured_alerts = processAlert(cpubreach, diskbreach, membreach)
        print(structured_alerts)
    """
    metrics = {'CPU': cpubreach, 'Memory': membreach, 'Disk': diskbreach} # Corrected order
    tobesent = {}
    try:
        for key, metric_df in metrics.items():
            if metric_df is not None and not metric_df.empty:
                user_alerts = {} # Group alerts by user within each metric
                metricName = 'CPUUsage' if key == 'CPU' else 'MemoryUsage' if key == 'Memory' else 'TotalFreeDiskGB'
                threshName = 'CPU_thresh' if key == 'CPU' else 'MEM_thresh' if key == 'Memory' else 'DISK_thresh'

                for _, row in metric_df.iterrows():
                    username = row['Username']
                    if username not in user_alerts:
                        user_alerts[username] = []

                    try:
                        # Safely evaluate Emails list
                        email_list = ast.literal_eval(row['Emails']) if isinstance(row['Emails'], str) else row['Emails']
                        email_list = email_list if isinstance(email_list, list) else []

                        # Safely evaluate IPAddress list
                        ip_address_list = ast.literal_eval(row['IPAddress_x']) if isinstance(row['IPAddress_x'], str) else row['IPAddress_x']
                        ip_address_list = ip_address_list if isinstance(ip_address_list, list) else [str(ip_address_list)] # Handle single IP case

                        # Ensure metric and threshold values are numeric
                        metric_val = pd.to_numeric(row[metricName], errors='coerce')
                        thresh_val = pd.to_numeric(row[threshName], errors='coerce')

                        if pd.isna(metric_val) or pd.isna(thresh_val):
                            logger.warning(f"Skipping alert for user {username} due to non-numeric metric/threshold.")
                            continue

                        alert_detail = {
                            "Hostname": row['Hostname'],
                            "IPAddress": ip_address_list,
                            "Metric": key,
                            "MetricValue": metric_val,
                            "Threshold": thresh_val, # Include threshold
                            "ManagementZone": row['ManagementZone'],
                            "DriveLetter": row['DriveLetter'],
                            "LogTimestamp": row['LogTimestamp'],
                            "Emails": email_list,
                            "UseAI": row.get('Alerting_AI', None) # Get AI preference
                        }
                        user_alerts[username].append(alert_detail)

                    except (ValueError, SyntaxError, TypeError) as e:
                        logger.warning(f"Could not parse Emails or IPAddress for user {username}: {e}. Row: {row}")
                        print(f"Could not parse Emails or IPAddress for user {username}: {e}. Row: {row}")
                    except KeyError as e:
                        logger.error(f"Missing expected column during alert processing for user {username}: {e}. Row: {row}")
                        print(f"Missing expected column during alert processing for user {username}: {e}. Row: {row}")
                    except Exception as e:
                        logger.error(f"Unexpected error processing alert for user {username}: {e}. Row: {row}", exc_info=True)
                        print(f"Unexpected error processing alert for user {username}: {e}. Row: {row}", exc_info=True)

                if user_alerts: # Only add metric key if there are alerts for it
                    tobesent[key] = user_alerts

    except Exception as e:
        logger.error(f"An error occurred during processAlert: {e}", exc_info=True)

    return tobesent


# def emailAlert_(to_emails, subject, body): # gmail
#     """
#     Sends an email to multiple recipients.
#     Args:
#         to_emails (list): A list of email addresses to send the email to.
#         subject (str): The subject of the email.
#         body (str): The body content of the email.
#     """
#     smtp_server = 'smtp.gmail.com'
#     port = 587
#     from_email = 'ncgalertsystem@gmail.com'
#     app_password = 'vwtq ztza vgzl hyxp'  
    
#     # Check if its an instance of a list. if not, make it a list 
#     if not isinstance(to_emails, list):
#         to_emails = [to_emails]

#     # Set up the MIMEMultipart object
#     msg = MIMEMultipart('alternative')
#     msg['From'] = from_email
#     msg['To'] = ", ".join(to_emails)  # Join all recipients into a single string
#     msg['Subject'] = subject

#     # Attach the email body
#     msg.attach(MIMEText(body, 'html'))

#     # Send the email
#     try:
#         with smtplib.SMTP(smtp_server, port) as server:
#             server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
#             server.login(from_email, app_password)  # Log in to your email account

#             # Send the email to all recipients
#             server.sendmail(from_email, to_emails, msg.as_string())
#     except Exception as e:
#         print(f"Error: {e}")

def emailAlert_(to_emails, subject, body): # zoho
    """
    Sends an email to multiple recipients.
    Args:
        to_emails (list): A list of email addresses to send the email to.
        subject (str): The subject of the email.
        body (str): The body content of the email.
    """
    SENDER_NAME = "NCG Alerting System"
    EMAIL_USER = 'ncgalertsystem@ncgafrica.com'
    
    smtp_server = 'smtp.zoho.com'
    port = 465
    from_email = 'ncgalertsystem@ncgafrica.com'
    app_password = 'tN5xghgufYE0'  
    
    # Check if its an instance of a list. if not, make it a list 
    if not isinstance(to_emails, list):
        to_emails = [to_emails]

    # Set up the MIMEMultipart object
    msg = MIMEMultipart('alternative')
    # msg['From'] = from_email
    msg["From"] = formataddr((SENDER_NAME, EMAIL_USER))
    msg['To'] = ", ".join(to_emails)  # Join all recipients into a single string
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'html'))

    try:
        # Connect to Zoho's SMTP server
        with smtplib.SMTP_SSL(smtp_server, port) as server:
            server.login(from_email, app_password)
            # Use sendmail instead of send_message for better compatibility
            server.sendmail(from_email, to_emails, msg.as_string())
            print("Email sent successfully!")
            logger.info('Email sent successfully!')
    except Exception as e:
        print("❌ Failed to send email:", str(e))
        logger.error(f'Failed to send email: {str(e)}')



def format_and_send_alert_email(container):
    """
    Formats and sends alert emails based on the processed alert container.
    Groups alerts by user before sending.
    """
    # Restructure alerts by user first
    mails =[]
    alerts_by_user = {}
    for metric_type, users_data in container.items():
        for username, alerts_list in users_data.items():
            if username not in alerts_by_user:
                alerts_by_user[username] = {'CPU': [], 'Memory': [], 'Disk': [], 'Emails': set()}
            alerts_by_user[username][metric_type].extend(alerts_list)
            # Collect all unique emails for this user across all metrics
            for alert in alerts_list:
                alerts_by_user[username]['Emails'].update(alert.get('Emails', []))

    # Iterate through users and send one email per user
    for username, user_alert_data in alerts_by_user.items():
        email_list = list(user_alert_data['Emails'])
        if not email_list:
            continue

        total_alerts = len(user_alert_data['CPU']) + len(user_alert_data['Memory']) + len(user_alert_data['Disk'])
        if total_alerts == 0:
            continue # Should not happen if container wasn't empty, but safety check

        # --- Format HTML Body ---
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Infrastructure Alert for {username}</title>
            <style>
                body {{ font-family: sans-serif; line-height: 1.6; }}
                .container {{ padding: 20px; border: 1px solid #ccc; }}
                h2 {{ color: #d9534f; }} /* Reddish */
                h3 {{ color: #5bc0de; }} /* Bluish */
                ul {{ list-style-type: none; padding-left: 0; }}
                li {{ margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                strong {{ color: #555; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Infrastructure Alert for {username}</h2>
                <p>High resource utilization detected:</p>
        """

        for metric_type in ['CPU', 'Memory', 'Disk']:
            alerts = user_alert_data[metric_type]
            if alerts:
                html_content += f"<h3>High {metric_type} Usage:</h3><ul>"
                for alert in alerts:
                    # Format IP addresses
                    ip_str = ', '.join(alert.get('IPAddress', []))
                    # Format metric value appropriately
                    value_format = ".2f" if metric_type != 'Disk' else ".1f" # Disk might be GB
                    metric_unit = "%" if metric_type != 'Disk' else "GB Free" # Clarify Disk threshold meaning
                    threshold_unit = "%" if metric_type != 'Disk' else "GB"

                    html_content += f"""
                        <li>
                            <strong>Server:</strong> {alert['Hostname']} ({ip_str})<br>
                            <strong>Zone:</strong> {alert['ManagementZone']} | <strong>Drive:</strong> {alert['DriveLetter']}<br>
                            <strong>Value:</strong> {alert['MetricValue']:{value_format}} {metric_unit} | <strong>Threshold:</strong> {'<' if metric_type == 'Disk' else '>'} {alert['Threshold']:{value_format}} {threshold_unit}<br>
                            <strong>Time:</strong> {alert['LogTimestamp']}
                        </li>
                    """
                html_content += "</ul>"

        # --- Add AI Analysis (Placeholder) ---
        # ai_analysis = get_ai_analysis(user_alert_data) # Implement this if needed
        # html_content += f"<h3>AI Analysis & Recommendations:</h3><div>{ai_analysis}</div>"

        html_content += f"""
                <p><strong>Total Alerts:</strong> {total_alerts}</p>
                <p><i>Alert generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i></p>
            </div>
        </body>
        </html>
        """
        # --- Send Email ---
        subject = f"ALERT: High Resource Usage Detected for {username} ({total_alerts} issues)"
        emailAlert_(to_emails=email_list, subject=subject, body=html_content)
        mails.append(email_list)
    outMails = ''
    for sublist in mails:
        for mail in sublist:
            outMails += mail + ', '

    if outMails != '':
        print(f'Notifications sent to {outMails}\n')
    else:
        print('No threshold breach detected')

# def format_and_send_alert_email(container):
#     """
#     This function formats and sends alert emails for high resource utilization (CPU, Disk, Memory) to the relevant users. 
#     It processes the structured alert data, formats it into an HTML email, and sends it to the recipients.

#     Steps:
#     1. **Iterate Through Metrics**:
#        - Loops through each metric type (CPU, Disk, Memory) in the `container` dictionary.

#     2. **Iterate Through Users**:
#        - For each metric type, iterates through the users and their associated alerts.

#     3. **Format Alert Details**:
#        - Formats the alert details for each user into an HTML structure, including server details, metric values, and timestamps.

#     4. **Compose Email Body**:
#        - Constructs the email body using the formatted alert details and additional information.

#     5. **Extract Recipient Emails**:
#        - Extracts unique email addresses from the alerts for each user.

#     6. **Send Email**:
#        - Sends the formatted email to the extracted email addresses using the `emailAlert_()` function.

#     Args:
#         container (dict): A nested dictionary containing structured alert data, grouped by metric type and user.

#     Example Input:
#         {
#             'CPU': {
#                 'user1': [
#                     {
#                         "Hostname": "server1",
#                         "IPAddress": ["192.168.1.1"],
#                         "Metric": "CPU",
#                         "MetricValue": 95,
#                         "ManagementZone": "Zone1",
#                         "DriveLetter": "C",
#                         "LogTimestamp": "2025-04-17 10:00:00",
#                         "Emails": ["user1@example.com"]
#                     },
#                     ...
#                 ],
#                 ...
#             },
#             'Disk': { ... },
#             'Memory': { ... }
#         }

#     Dependencies:
#         - `emailAlert_()`: Sends the email to the recipients.
#         - `datetime`: Used to include the current timestamp in the email.

#     Example Usage:
#         format_and_send_alert_email(outwardSendingConstructor)
#     """
#     date = datetime.now()
#     for metric_type, users in container.items():
#         count_of_servers = 0
#         for user, alerts in users.items():
#             count_of_servers += len(alerts)

#             def format_list_section(alerts_list):
#                 """Helper function to format a list of servers into a string."""
#                 if not alerts_list:
#                     return f"<p>No servers are currently experiencing high {metric_type} usage.</p>"
#                 else:
#                     section_body = f"<p>The following servers are currently experiencing high {metric_type} usage:</p><ul>"
#                     for alert in alerts_list:
#                         # Format each alert using the structured data
#                         formatted_server = (
#                             f"{alert['Hostname']}_{alert['IPAddress']}, "
#                             f"{metric_type}({alert['MetricValue']}), "
#                             f"MgtZone({alert['ManagementZone']}), "
#                             f"Drive({alert['DriveLetter']}), "
#                             f"{alert['LogTimestamp']}, "
#                         )
#                         section_body += f"<li>{formatted_server}</li>"
#                     section_body += "</ul>"
#                     return section_body

#             # Format the email body
#             entry_section = format_list_section(alerts)
#             body = f"""
#             <!DOCTYPE html>
#             <html>
#             <head>
#                 <title>Infrastructure Alert On {metric_type} Usage For {users}</title>
#                 <style>
#                     body {{
#                         font-family: Arial, sans-serif;
#                         font-size: 14px;
#                         color: #333;
#                         background-color: #f4f4f4;
#                         padding: 20px;
#                     }}
#                     .container {{
#                         background-color: #fff;
#                         padding: 20px;
#                         border-radius: 5px;
#                         box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
#                     }}
#                     h2 {{
#                         color: #d9534f;
#                         margin-top: 0;
#                         margin-bottom: 10px;
#                     }}
#                     p {{
#                         margin-bottom: 10px;
#                     }}
#                     ul {{
#                         list-style-type: disc;
#                         margin-left: 20px;
#                     }}
#                     li {{
#                         margin-bottom: 5px;
#                     }}
#                     .info {{
#                         margin-top: 20px;
#                         font-size: 0.9em;
#                         color: #777;
#                         border-top: 1px solid #ddd;
#                         padding-top: 10px;
#                     }}
#                     .footer {{
#                         margin-top: 20px;
#                         font-size: 0.9em;
#                         color: #777;
#                         border-top: 1px solid #ddd;
#                         padding-top: 10px;
#                         text-align: center;
#                     }}
#                 </style>
#             </head>
#             <body>
#                 <div class="container">
#                     <p>Dear Operations Team,</p>
#                     <p>This is an <strong>automated alert</strong> on high resource utilization detected on some servers.</p>
#                     <h2>High {metric_type} Usage:</h2>
#                     {entry_section}
#                     <div class="info">
#                         <p><strong>Additional Information:</strong></p>
#                         <ul>
#                             <li>This alert was generated at: {date}</li>
#                             <li>The data was gathered from the last available logs.</li>
#                             <li>For further details, please refer to the Infrastructure Monitoring Dashboard.</li>
#                         </ul>
#                     </div>
#                     <div class="footer">
#                         <p>Thank you,</p>
#                         <p>The Infrastructure Monitoring System</p>
#                     </div>
#                 </div>
#             </body>
#             </html>
#             """

#             # Extract unique emails from the alerts
#             emails = set()
#             for alert in alerts:
#                 emails.update(alert["Emails"])
#             email_list = list(emails)

#             # Send the email
#             subject = f"ALERT: High {metric_type} Usage on {count_of_servers} Servers"
#             emailAlert_(to_emails=email_list, subject=subject, body=body)
#             print(f'Notification Sent to {email_list}')
