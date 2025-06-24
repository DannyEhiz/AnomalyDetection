from ServerSide.services.modelling import train_and_validate, loadData
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



def runModelling():
    latestTime = retrieveRecord('ServerSide/database/aux.json', 'latestLog', 'logTime')
    with sqlite3.connect('ServerSide/database/mini.sqlite3') as conn:
        latestLog = pd.read_sql_query('SELECT * FROM lastLog', conn)

    if latestLog.empty: #* It will be empty on the first run, so we skip running detector, and begin detector in the second run
        return None
    if latestTime == latestLog.LogTimestamp.max(): # Means data has not been updated
        return None
    updateRecord('ServerSide/database/aux.json', 'latestLog', 'logTime', latestLog.LogTimestamp.max(), append=False)

    for server in np.array(latestLog.Hostname.unique().tolist()).reshape(-1):
        model_path = f'ServerSide/models/LSTM/{server}/{server}_lstm.h5'
        scaler_path = f'ServerSide/models/scalers/{server}/{col}_scaler.pkl'
        with open(f'ServerSide/models/LSTM/{server}/thresholds.json', 'r') as f:
            thresholds = json.load(f)
        
        if not os.path.exists(model_path):
            if not os.path.exists(scaler_path):
                print(f"Seeing {server} the first time. No model or scaler found. Skipping...")
                logger.error(f"Seeing {server} the first time. No model or scaler found. Skipping...")
            else:
                print(f"🚫 Scaler exists for {server} but model is missing. \nData is being collected but model is not trained...Skipping")
                logger.error(f"Scaler exists for {server} but model is missing.\nData is being collected but model is not trained.. Skipping")
            continue

        with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
            c.execute(f"SELECT MAX(LogTimestamp) FROM Infra_Utilization WHERE Hostname = '{server}'")
            maxTime = c.fetchone()
            last2Hours = (pd.to_datetime(maxTime) - timedelta(hours=2)).strftime('%Y-%m-%s %H:%M:%S')
            query = f"""
                    SELECT * FROM Infra_Utilization WHERE Hostname = '{server}'
                    AND LogTimestamp > '{last2Hours[0]}'
                """
            data = pd.read_sql_query(query, conn)

        data = processModellingData(data)
        data = data.iloc[:60]  # select 60 time stamps

        for col in data.columns:
            scaler_path = f'ServerSide/models/scalers/{server}/{col}_scaler.pkl'
            transformer = joblib.load(scaler_path)
            data[col] = transformer.transform(data[[col]])

        model = load_model(model_path)
        sequence = np.array(data).reshape(1, 60, 6)
        reconstructed = model.predict(sequence)
        original_last_minute = sequence[0, -1, :]
        reconstructed_last_minute = reconstructed[0, -1, :] 
        reconstruction_error = np.mean((original_last_minute - reconstructed_last_minute) ** 2)
        if reconstruction_error >= thresholds['robust']:
            anomaly_level, anomalyTrue = 'Severe', True
        elif reconstruction_error >= thresholds['lenient']:
            anomaly_level,  anomalyTrue = 'High', True
        elif reconstruction_error >= thresholds['strict']:
            anomaly_level, anomalyTrue = 'Light', True
        else:
            anomalyTrue = False

        feature_names = ['Hour', 'CPUUsage', 'MemoryUsage', 'DiskUsage', 'NetworkTrafficReceived', 'NetworkTrafficSent']
        feature_errors = np.abs(original_last_minute - reconstructed_last_minute)
        # Create a DataFrame for easy tracking
        error_report = pd.DataFrame({
            'Feature': feature_names,
            'Error': feature_errors
        })
        error_report = error_report.sort_values(by='Error', ascending=False).reset_index(drop=True)
        error_report.to_csv(f'ServerSide/models/LSTM/{server}/featImportance.csv', index=False)



# def saveLastCollectionTime(data):
#     with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
#         c = conn.cursor()  
#         try:
#             if data is not None:
#                 data.to_sql('latestTime', conn, if_exists='replace', index=False) 
#                 gc.collect()
#                 del gc.garbage[:]
#             else:
#                 print('Couldnt save lastLog to db')
#                 pass
#         except Exception as e:
#             print(f"Error occurred while saving latest time to SQLite: {e}") 

# def upDatelastLogTime(serverName):
#     with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
#         c = conn.cursor()  
#         lastLogTime = pd.read_sql_query('SELECT * FROM latestTime', conn)
#         infraLog = pd.read_sql_query('SELECT * FROM Infra_Utilization LIMIT 1', conn)
#         infraLogEmpty =  infraLog.empty
#         lastestLogEmpty = lastLogTime.empty    


# def upDateLastLogTime():
#     with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
#         c = conn.cursor()  
#         lastLogTime = pd.read_sql_query('SELECT * FROM latestTime', conn)
#         infraLog = pd.read_sql_query('SELECT * FROM Infra_Utilization LIMIT 1', conn)
#         infraLogEmpty =  infraLog.empty
#         lastestLogEmpty = lastLogTime.empty

#         if infraLogEmpty:
#             return None
#         if lastestLogEmpty:
#             query = """
#                         SELECT Hostname, MAX(LogTimestamp) AS LatestLogTime
#                         FROM Infra_Utilization
#                         GROUP BY Hostname
#                     """
#             data = pd.read_sql_query(query, conn)
#             saveLastCollectionTime(data)
#             print('Updated latest log collection time')


# def dataIsUpdated(serverName):
#     with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
#         lastInfraLog = pd.read_sql_query(f"SELECT MAX(LogTimestamp) FROM Infra_Utilization WHERE Hostname='{serverName}'", conn).values.tolist()[0][0]
#         lastTimeKeptLog = pd.read_sql_query(f"SELECT MAX(LogTimestamp) FROM latestTime WHERE Hostname = '{serverName}'", conn).values.tolist()[0][0]

#     if lastInfraLog == lastTimeKeptLog:
#         return True 
#     else:
#         return False
    

    # with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
    #     c = conn.cursor()  
    #     servers = pd.read_sql_query('SELECT DISTINCT(Hostname) FROM Infra_Utilization', conn).values.tolist()
    #     servers = np.array(servers).flatten()
    

    # with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
    #     conn.execute('PRAGMA journal_mode=WAL')
    #     servers = pd.read_sql_query("SELECT DISTINCT Hostname FROM Infra_Utilization", conn)
    # server_list = servers['Hostname'].tolist()

    # print(f"Found {len(server_list)} servers in the database.")
    # for server in server_list:
    #     print(f"Processing server: {server}")
    #     data = loadData(server)
    #     if data is not None and not data.empty:
    #         try:
    #             accepted = train_and_validate(data, server, modelAcceptability='robust', seq_len=60)
    #             if accepted:
    #                 print(f"Model for {server} trained and saved successfully.")
    #                 logger.info(f"Model for {server} trained and saved successfully.")
    #             else:
    #                 print(f"Model for {server} was not accepted.")
    #                 logger.info(f"Model for {server} was not accepted.")
    #         except Exception as e:
    #             print(f"There was an error training model for {server}: {e}.")
    #             logger.error(f"There was an error training model for {server}: {e}.")
    #     else:
    #         print(f"No data found for server {server}. Skipping to next server...")
    #         logger.error(f"No data found for server {server}. Skipping to next server...")
    #     print(f'\nWating for 5secs for the next server run...')
    #     time.sleep(5) 