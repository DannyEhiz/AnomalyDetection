import sys 
import os
sys.path.append(os.path.abspath('.'))
import concurrent.futures
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings 
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from ServerSide.core.tinyDBHandler import retrieveRecord, updateRecord, createRecord, tableIsExisting
from ServerSide.core.createTables import featureImportance
from tensorflow.keras import Sequential, Input # type: ignore
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from datetime import datetime, timedelta
import csv, json
import xgboost as xgb
import shap
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, joblib, time
import logging
from logging.handlers import RotatingFileHandler
from ServerSide.core.logger import logging_setup
logger = logging_setup(log_dir='logs/modelling', 
                       general_log='modellingInfo.log', 
                       error_log='modellingError.log', 
                       loggerName='modellingLogger')

def loadData(serverName: str):
    """
    Load data from SQLite database for a specific server.
    Args:
        serverName (str): The name of the server to filter data by.
    Returns:
        pd.DataFrame: DataFrame containing the server's utilization data.
    """
    try:
        with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            data = pd.read_sql_query("SELECT * FROM Infra_Utilization WHERE Hostname = ?", conn, params=(serverName,))
        if not data.empty:
            data['LogTimestamp'] = pd.to_datetime(data['LogTimestamp'])
            data['NetworkTrafficReceived'] = round( data['NetworkTrafficReceived']/ 1_048_576, 1)
            data['NetworkTrafficSent'] = round( data['NetworkTrafficSent']/ 1_048_576, 1)
            data.set_index('LogTimestamp', inplace = True)
            data = data[[ 'CPUUsage', 'MemoryUsage', 'DiskUsage', 'NetworkTrafficReceived', 'NetworkTrafficSent']]
            data = data.resample('1min', closed='right', label='right').last()
            data['Hour'] = data.index.to_series().dt.hour
            data.dropna(inplace = True)
            # data = data[[ 'CPUUsage', 'MemoryUsage', 'DiskUsage', 'NetworkTrafficReceived', 'NetworkTrafficSent', 'Hour']]
            return data
        else:
            return None
    except Exception as e:
        print(f"Error from loadData function in modelling: loading data for server {serverName} encountered error: {e}")
        return None
    
def preprocessData(df: pd.DataFrame, serverName: str):
    """
    Preprocess the DataFrame by removing outliers and scaling the data.
    Args:
        df (pd.DataFrame): DataFrame containing the server's utilization data.
    Returns:
        pd.DataFrame: Preprocessed DataFrame with outliers removed and scaled.
    """
    if df is not None or not df.empty:
        # if len(df) < 4320:
        #     print(f'{serverName} has less than 3days worth of data. Not enough to train on. Skipping')
        #     logger.error(f'{serverName} has less than 3days worth of data. Not enough to train on. Skipping')
        #     return None
        cols = [ 'CPUUsage', 'MemoryUsage', 'DiskUsage', 'NetworkTrafficReceived', 'NetworkTrafficSent', 'Hour']
        try:
            data = df.copy()
            for col in cols:
                os.makedirs(f'ServerSide/models/scalers/{serverName}', exist_ok=True)
                scaler = MinMaxScaler()
                data[col] = scaler.fit_transform(data[[col]])
                joblib.dump(scaler, open(f"ServerSide/models/scalers/{serverName}/{col}_scaler.pkl", 'wb'))

            print(f"Data for server {serverName} preprocessed successfully.")    
            logger.info(f"Data for server {serverName} preprocessed successfully.")    
            return data
        except Exception as e:
            print(f"Error from preprocessData function in modelling while trying to preprocess {serverName} data: {e}")
            return None
    else:
        print(f'Empty Dataframe for {serverName}. Couldnt process data')
        logger.error(f'Empty dataframe for {serverName}.Couldnt process data')
        return None


def createSequence(df: pd.DataFrame, seq_len):
    X_train = np.array([df[i:i+seq_len] for i in range(len(df)-seq_len)])
    return X_train


def buildLSTM_Autoencoder(sequence_length):
    """
    Build and compile the LSTM model.
    Args:
        input_shape (tuple): Shape of the input data.
    Returns:
        keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential([
        Input(shape=(sequence_length, 6)),
        LSTM(128, activation='relu', return_sequences=False),
        RepeatVector(sequence_length),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(6))
    ])
    model.compile(optimizer='adam', loss='mse')
    # model.fit(data, data, epochs=4, batch_size=32, validation_split=0.1, verbose=1 ) # type: ignore
    
    return model

def train_and_validate(data, serverName, seq_len=60, modelAcceptability='robust', val_split=0.1, epochs=20, batch_size=16):
    """
    modelAcceptability is either strict, lenient, or robust
    Full training pipeline with dynamic thresholding
    LSTM Autoencoder outputs reconstruction error per sample.
    XGBoost is trained to predict that error from the last row of each input sequence.
    SHAP explains the XGBoost predictions → you get feature importance per anomaly. And its saved
    Returns: modelAccepted, model, threshold, historyLoss, historyValLoss, fig
    """

    if modelAcceptability.lower() == 'strict':
        accept = 1.5
    elif modelAcceptability.lower() == 'lenient':
        accept = 2.0
    elif modelAcceptability.lower() == 'robust':
        accept = 3.0
    else:
        accept = 3.0
        

    scaled_data = preprocessData(data, serverName=serverName)
    if scaled_data is None or scaled_data.empty:
        print(f'Data passed in for {serverName} is None or empty. Exiting training for {serverName}')
        logger.error(f'Data passed in for {serverName} is None or empty. Exiting training for {serverName}')
        return None
    
    X_train = createSequence(scaled_data, seq_len)
    
    # X_train, X_test = train_test_split(sequences, test_size=test_size, shuffle=False)
    
    # 3. Build and train model
    try:
        model = buildLSTM_Autoencoder(seq_len)
        history = model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=[EarlyStopping(patience=3)],
            verbose=1 # type: ignore
        )
    except Exception as e:
        print(f'Error in training model for {serverName}: {e}')
        logger.error(f'Error training model for {serverName}: {e}')
        return None

    historyLoss = history.history['loss']
    historyValLoss = history.history['val_loss']

    # 5. Calculate dynamic threshold (using validation reconstructions)
    val_X = X_train[:int(len(X_train)*val_split)] #use 10% for validation
    val_recon = model.predict(val_X)  # Get the actual validation set
    val_errors = np.mean(np.square(val_X - val_recon), axis=(1,2))
    threshold = np.percentile(val_errors, 99) 
    
    print(f"\nTraining complete on {serverName}. Recommended anomaly threshold: {threshold:.6f}\n")
    print(f"Final Training Loss: {history.history['loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.6f}")
    
    # 6. Model acceptance check
    max_allowed_loss = np.mean(val_errors) + accept * np.std(val_errors)
    final_val_loss = history.history['val_loss'][-1]
    
    if final_val_loss > max_allowed_loss:
        print(f"\t\t❌ {serverName} Model rejected - Validation loss exceeds threshold ({final_val_loss:.6f} > {max_allowed_loss:.6f})")
        logger.error(f"{serverName} Model rejected - Validation loss exceeds threshold ({final_val_loss:.6f} > {max_allowed_loss:.6f})")
        modelAccepted = False
        with open('ServerSide/models/rejectedModels.csv', 'a', newline='') as csvfile:  # Open in append mode
            csv_writer = csv.writer(csvfile)
            # Write header if the file is empty
            if csvfile.tell() == 0:
                csv_writer.writerow(['Model Name', 'Max Allowed Loss', 'Validation Loss'])  # Header
            csv_writer.writerow([f'{serverName}_lstm.h5',f'{max_allowed_loss:.6f}', f'{final_val_loss:.6f}'])
        return modelAccepted

    print(f"\t\t✅ {serverName} Model accepted - Validation loss within acceptable range")
    logger.info(f"{serverName} Model accepted - Validation loss within acceptable range")
    modelAccepted = True

    strictThresh = np.mean(val_errors) + 1.5 * np.std(val_errors)
    lenientThresh = np.mean(val_errors) + 2.0 * np.std(val_errors)
    robustThresh = np.mean(val_errors) + 3.0 * np.std(val_errors)
    thresholds = {
        'strict': strictThresh,
        'lenient': lenientThresh,
        'robust': robustThresh
    }
    os.makedirs(f'ServerSide/models/LSTM/{serverName}', exist_ok=True)
    model.save(f'ServerSide/models/LSTM/{serverName}/{serverName}_lstm.h5')
    with open(f'ServerSide/models/LSTM/{serverName}/thresholds.json', 'w') as f:
        json.dump(thresholds, f)

    saveTrainingHistory(serverName)


    # Summarize the loss and fit a surrogate model like XGBoost. That gives you interpretable SHAP values explaining which features contributed most to high reconstruction error (i.e., anomaly score).
    reconstruction = model.predict(X_train)
    recon_error = np.mean((reconstruction - X_train)**2, axis=(1, 2))
    X_flat = X_train[:, -1, :]  # Use the last timestep of each sequence
    regressor = xgb.XGBRegressor()
    regressor.fit(X_flat, recon_error)
    # SHAP on surrogate
    explainer = shap.Explainer(regressor)
    shap_values = explainer(X_flat)
    shap_array = np.abs(shap_values.values)  
    mean_shap = np.mean(shap_array, axis=0)  

    # Create a ranked DataFrame of feature that most contribute to the anomaly
    importance_df = pd.DataFrame({
        'Feature': [ 'CPUUsage', 'MemoryUsage', 'DiskUsage', 'NetworkTrafficReceived', 'NetworkTrafficSent', 'Hour'],
        'Mean_SHAP_Importance': mean_shap
    }).sort_values(by='Mean_SHAP_Importance', ascending=False).reset_index(drop=True)
    
    saveFeatureImportance(serverName, importance_df)
    # importance_df.to_csv(f'ServerSide/models/LSTM/{serverName}/featImportance.csv', index=False)


    shap.summary_plot(shap_values, X_flat, feature_names=['CPUUsage', 'MemoryUsage', 'DiskUsage', 'NetworkTrafficReceived', 'NetworkTrafficSent', 'Hour'], show=False)
    plotfig = plt.gcf()
    os.makedirs(f'ServerSide/models/shapSummaryPlot', exist_ok=True)
    plotfig.savefig(f"ServerSide/models/shapSummaryPlot/{serverName}_@shap_summary.svg", bbox_inches="tight")
    plt.close(plotfig)

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(val_errors))),
        y=val_errors,
        mode='lines',
        name='Reconstruction Loss (MSE)',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(val_errors))),
        y=[threshold] * len(val_errors),
        mode='lines',
        name='Anomaly Threshold',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title=f"Reconstruction Loss (MSE) - {serverName}",
        xaxis_title="Sequence Index",
        yaxis_title="Loss",
        legend=dict(x=0, y=1),
        template='plotly_white'
    )
    os.makedirs(f'ServerSide/models/reconLossPlot', exist_ok=True)
    fig.write_image(f'ServerSide/models/reconLossPlot/{serverName}.svg', scale=2)

    return  modelAccepted 

# A good mse, in a normalized data, should range between 0.0001 -> 0.01 

def saveFeatureImportance(serverName, data):
    """Save the feature importance of the model. Used to know what feature is causing the anomaly"""
    featureImportance(serverName)
    with sqlite3.connect('ServerSide/database/featImportance.sqlite3') as conn:
        c = conn.cursor()  
        data.to_sql(serverName, conn, if_exists='replace', index=False) 
        del data


def saveTrainingHistory(server):
    with sqlite3.connect('ServerSide/database/mini.sqlite3') as conn:
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO trainHistory (server, lastTrained)
            VALUES (?, ?)
        """, (server, datetime.now()))
        conn.commit()   




# def runTrainer():
#     if os.path.isfile('ServerSide/database/main.sqlite3'):
#         with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
#             c = conn.cursor()
#             c.execute('SELECT DISTINCT(Hostname) FROM Infra_Utilization')
#             servers = c.fetchall()
#         servers = np.array(servers).reshape(-1).tolist()
#         recentSeen = False

#         if os.path.isfile('ServerSide/database/mini.sqlite3'):
#             with sqlite3.connect('ServerSide/database/mini.sqlite3') as conn:
#                 c = conn.cursor()
#                 c.execute("SELECT DISTINCT(Hostname) FROM lastLog")
#                 recentDataServers = c.fetchall()
#                 recentDataServers = np.array(recentDataServers).reshape(-1).tolist()
#                 recentSeen = True
        
#         serversToTrain = servers
#         if not tableIsExisting('ServerSide/database/aux.json', 'serversToTrain'):
#             createRecord('ServerSide/database/aux.json', 'serversToTrain', 'servers')
#         updateRecord('ServerSide/database/aux.json', 'serversToTrain', 'servers', serversToTrain, append = False)

#         if recentSeen:
#             serversToTrain = [i for i in servers if i in recentDataServers] # type: ignore
        
#         for hosts in serversToTrain:
#             data = loadData(hosts)
#             if data is not None:
#                 train_and_validate(data, hosts)
#             else:
#                 print(f'No data for {hosts} to train on')



# import concurrent.futures

def runTrainer():
    if os.path.isfile('ServerSide/database/main.sqlite3'):
        with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
            c = conn.cursor()
            c.execute('SELECT DISTINCT(Hostname) FROM Infra_Utilization')
            servers = c.fetchall()
        servers = np.array(servers).reshape(-1).tolist()
        recentSeen = False

        if os.path.isfile('ServerSide/database/mini.sqlite3'):
            with sqlite3.connect('ServerSide/database/mini.sqlite3') as conn:
                c = conn.cursor()
                c.execute("SELECT DISTINCT(Hostname) FROM lastLog")
                recentDataServers = c.fetchall()
                recentDataServers = np.array(recentDataServers).reshape(-1).tolist()
                recentSeen = True
        
        serversToTrain = servers
        if recentSeen:
            serversToTrain = [i for i in servers if i in recentDataServers]  # type: ignore

        def train_one_server(host):
            print(f"🚀 Starting training for {host}...")
            data = loadData(host)
            if data is not None:
                train_and_validate(data, host)
                return f"✅ Training completed for {host}"
            else:
                return f"⚠️ No data for {host}, skipping."

        # Run training in parallel with progress tracking
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(train_one_server, server): server for server in serversToTrain}

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(result)

        print("🎉 All training jobs completed!")

            

if __name__ == "__main__":
    servers = retrieveRecord('ServerSide/database/aux.json', 'serversToTrain', 'servers')
    print(f'starting training sequence for seen servers: \n {servers}')
    runTrainer()