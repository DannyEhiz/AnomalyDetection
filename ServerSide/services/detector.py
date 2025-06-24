import pandas as pd 
import numpy as np
import sqlite3
from datetime import datetime, timedelta, time
import os, gc, joblib, json
from ServerSide.core.tinyDBHandler import retrieveRecord, updateRecord, createRecord, tableIsExisting, removeItemFromRecord
from ServerSide.core.createTables import generateAutoID
from tensorflow.keras.models import load_model
from logging.handlers import RotatingFileHandler
from ServerSide.core.logger import logging_setup
logger = logging_setup(log_dir='logs/modelling', 
                       general_log='modellingInfo.log', 
                       error_log='modellingError.log', 
                       loggerName='modellingLogger')

def processModellingData(data):
    data['LogTimestamp'] = pd.to_datetime(data['LogTimestamp'])
    data['NetworkTrafficReceived'] = data['NetworkTrafficReceived']/ 1024
    data['NetworkTrafficSent'] = data['NetworkTrafficSent']/ 1024
    data = data.resample('1min', on='LogTimestamp', closed='right', label='right').mean()
    data.dropna(inplace = True)
    data['Hour'] = data.index.hour
    return data


def saveAnomaly(metric, severity, serverName, timeStamp, ai_summary):
    with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
        conn.execute('PRAGMA journal_mode=WAL;')
        c = conn.cursor()
        c.execute(
            '''
            INSERT INTO anomalies (ID, METRIC, SEVERITY, SOURCE, TIMESTAMP, AI_SUMMARY)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (generateAutoID(), metric, severity, serverName, timeStamp, ai_summary)
        )
        conn.commit()


def process_server(server, confirmedLatestLog):
    """
    Processes a single server for anomaly detection, including loading models, scalers,
    running predictions, and saving anomalies if detected.
    """
    model_path = f'ServerSide/models/LSTM/{server}/{server}_lstm.h5'
    thresholds_path = f'ServerSide/models/LSTM/{server}/thresholds.json'

    if not os.path.exists(model_path) or not os.path.exists(thresholds_path):
        print(f"🚫 Missing model or thresholds for {server}. Skipping...")
        logger.error(f"Missing model or thresholds for {server}. Skipping...")
        return

    with open(thresholds_path, 'r') as f:
        thresholds = json.load(f)

    with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
        c = conn.cursor()
        c.execute(f"SELECT MAX(LogTimestamp) FROM Infra_Utilization WHERE Hostname = '{server}'")
        maxTime = c.fetchone()[0]
        last2Hours = (pd.to_datetime(maxTime) - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
        query = f"""
                SELECT * FROM Infra_Utilization WHERE Hostname = '{server}'
                AND LogTimestamp > '{last2Hours}'
            """
        data = pd.read_sql_query(query, conn)

    data = processModellingData(data).iloc[:60]
    for col in data.columns:
        scaler_path = f'ServerSide/models/scalers/{server}/{col}_scaler.pkl'
        if not os.path.exists(scaler_path):
            print(f"🚫 Missing scaler for {server} - {col}. Skipping...")
            logger.error(f"Missing scaler for {server} - {col}. Skipping...")
            return

        transformer = joblib.load(scaler_path)
        data[col] = transformer.transform(data[[col]])

    model = load_model(model_path)
    sequence = np.array(data).reshape(1, 60, 6)
    reconstructed = model.predict(sequence)

    original_last_minute = sequence[0, -1, :]
    reconstructed_last_minute = reconstructed[0, -1, :]
    reconstruction_error = np.mean((original_last_minute - reconstructed_last_minute) ** 2)

    anomalyTrue = False
    if reconstruction_error >= thresholds['robust']:
        anomaly_level = 'Severe'
        anomalyTrue = True
    elif reconstruction_error >= thresholds['lenient']:
        anomaly_level = 'High'
        anomalyTrue = True
    elif reconstruction_error >= thresholds['strict']:
        anomaly_level = 'Light'
        anomalyTrue = True

    if anomalyTrue:
        feature_names = ['Hour', 'CPUUsage', 'MemoryUsage', 'DiskUsage', 'NetworkTrafficReceived', 'NetworkTrafficSent']
        feature_errors = np.abs(original_last_minute - reconstructed_last_minute)
        error_report = pd.DataFrame({
            'Feature': feature_names,
            'Error': feature_errors
        }).sort_values(by='Error', ascending=False).reset_index(drop=True)

        error_report.to_csv(f'ServerSide/models/LSTM/{server}/featImportance.csv', index=False)

        ai_summary_path = f'ServerSide/models/LSTM/{server}/{server}_aiSummary.txt'
        with open(ai_summary_path, 'w') as f:
            f.write(f"Anomaly Level: {anomaly_level}\n")
            f.write(f"Top contributing feature: {error_report['Feature'].iloc[0]}\n")

        saveAnomaly(error_report['Feature'].iloc[0], anomaly_level, server, confirmedLatestLog, ai_summary_path)
        print(f"🚨 Anomaly Detected for {server}: {anomaly_level}")
        logger.info(f'Anomaly Detected for {server}: {anomaly_level}')


def runModelling():
    """
    Runs the anomaly detection process on the most recently refreshed telemetry data.
    """
    latestTime = retrieveRecord('ServerSide/database/aux.json', 'latestLog', 'logTime')
    try:
        with sqlite3.connect('ServerSide/database/mini.sqlite3') as conn:
            latestLog = pd.read_sql_query('SELECT * FROM lastLog', conn)
    except Exception as e:
        latestLog = pd.DataFrame()

    if latestLog.empty or latestTime == latestLog.LogTimestamp.max():
        return None

    updateRecord('ServerSide/database/aux.json', 'latestLog', 'logTime', latestLog.LogTimestamp.max(), append=False)
    confirmedLatestLog = retrieveRecord('ServerSide/database/aux.json', 'latestLog', 'logTime')

    for server in np.array(latestLog.Hostname.unique().tolist()).reshape(-1):
        process_server(server, confirmedLatestLog)