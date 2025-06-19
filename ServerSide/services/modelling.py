import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import plotly.graph_objects as go
import sqlite3
import os, joblib
import logging
from logging.handlers import RotatingFileHandler
from core.logger import logging_setup
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
        with sqlite3.connect('main.sqlite3') as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            data = pd.read_sql_query(f"SELECT * FROM Infra_Utilization WHERE Hostname = ?", conn, params=(serverName,))
        if not data.empty:
            data['LogTimestamp'] = pd.to_datetime(data['LogTimestamp'])
            data['NetworkTrafficReceived'] = data['NetworkTrafficReceived']/ 1024
            data['NetworkTrafficSent'] = data['NetworkTrafficSent']/ 1024
            data = data[data.LogTimestamp >= '2025-06-11']
            data = data.resample('1min', on='LogTimestamp', closed='right', label='right').mean()
            data.dropna(inplace = True)
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
        cols = ['CPUUsage', 'MemoryUsage', 'DiskUsage', 'NetworkTrafficReceived', 'NetworkTrafficSent']
        try:
            data = df.copy()
            for col in cols:
                os.makedirs(f'models/scalers/{serverName}', exist_ok=True)
                scaler = MinMaxScaler()
                data[col] = scaler.fit_transform(data[[col]])
                joblib.dump(scaler, open(f"models/scalers/{serverName}/{col}_scaler.pkl", 'wb'))

            print(f"Data for server {serverName} preprocessed successfully.")    
            logger.info(f"Data for server {serverName} preprocessed successfully.")    
            return data
        except Exception as e:
            print(f"Error from preprocessData function in modelling while trying to preprocess {serverName} data: {e}")
            return None

def createSequence(df: pd.DataFrame, seq_len):
    X_train = np.array([df[i:i+seq_len] for i in range(len(df)-seq_len)])
    return X_train


def buildLSTM_Autoencoder(data, sequence_length):
    """
    Build and compile the LSTM model.
    Args:
        input_shape (tuple): Shape of the input data.
    Returns:
        keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(sequence_length, 5), return_sequences=False),
        RepeatVector(sequence_length),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(5))
    ])

    model.compile(optimizer='adam', loss='mse')
    # model.fit(data, data, epochs=4, batch_size=32, validation_split=0.1, verbose=1 ) # type: ignore
    
    return model

def train_and_validate(data, serverName, seq_len, val_split=0.1, epochs=20, batch_size=32):
    """
    Full training pipeline with dynamic thresholding
    Returns: model, train_history, threshold
    """
    scaled_data = preprocessData(data, serverName=serverName)
    if scaled_data is None or scaled_data.empty:
        return None
    
    X_train = createSequence(scaled_data, seq_len)
    
    # X_train, X_test = train_test_split(sequences, test_size=test_size, shuffle=False)
    
    # 3. Build and train model
    model = buildLSTM_Autoencoder(X_train, seq_len)
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[EarlyStopping(patience=3)],
        verbose=1 # type: ignore
    )
    
    historyLoss = history.history['loss']
    historyValLoss = history.history['val_loss']
    
    # 5. Calculate dynamic threshold (using validation reconstructions)
    val_recon = model.predict(X_train[:int(len(X_train)*val_split)])  # Get the actual validation set
    val_errors = np.mean(np.square(X_train[:int(len(X_train)*val_split)] - val_recon), axis=(1,2))
    threshold = np.percentile(val_errors, 99) 
    
    print(f"\nTraining complete. Recommended anomaly threshold: {threshold:.6f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.6f}")
    
    # 6. Model acceptance check
    max_allowed_loss = np.mean(val_errors) + 2 * np.std(val_errors)
    final_val_loss = history.history['val_loss'][-1]
    
    if final_val_loss <= max_allowed_loss:
        print(f"\t\t✅ {serverName} Model accepted - Validation loss within acceptable range")
        logger.info(f"{serverName} Model accepted - Validation loss within acceptable range")
        modelAccepted = True
        model.save(f'models/LSTM/{serverName}_lstm.h5')
    else:
        print(f"\t\t❌ {serverName} Model rejected - Validation loss exceeds threshold ({final_val_loss:.6f} > {max_allowed_loss:.6f})")
        logger.error(f"{serverName} Model rejected - Validation loss exceeds threshold ({final_val_loss:.6f} > {max_allowed_loss:.6f})")
        modelAccepted = False
        with open('models/rejectedModels.csv', 'a', newline='') as csvfile:  # Open in append mode
            csv_writer = csv.writer(csvfile)
            # Write header if the file is empty
            if csvfile.tell() == 0:
                csv_writer.writerow(['Model Name', 'Max Allowed Loss', 'Validation Loss'])  # Header
            csv_writer.writerow([f'{serverName}_lstm.h5',f'{max_allowed_loss:.6f}', f'{final_val_loss:.6f}'])

    
    mean_mse = np.mean(val_errors)
    std_mse = np.std(val_errors)
    anomaly_threshold = mean_mse + 3 * std_mse

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
        y=[anomaly_threshold] * len(val_errors),
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

    return modelAccepted, model, threshold, historyLoss, historyValLoss, fig


if "__name__" == "__main__":
    with sqlite3.connect('main.sqlite3') as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        servers = pd.read_sql_query("SELECT DISTINCT Hostname FROM Infra_Utilization", conn)
    server_list = servers['Hostname'].tolist()

    print(f"Found {len(server_list)} servers in the database.")
    for server in server_list:
        print(f"Processing server: {server}")
        data = loadData(server)
        if data is not None and not data.empty:
            result = train_and_validate(data, server, seq_len=60)
            if result is not None:
                modelAccepted, model, threshold, historyLoss, historyValLoss, fig = result
                if modelAccepted:
                    print(f"Model for {server} trained and saved successfully.")
                else:
                    print(f"Model for {server} was not accepted.")
            else:
                print(f"Training failed for server {server}.")
        else:
            print(f"No data found for server {server}. Skipping to next server...")