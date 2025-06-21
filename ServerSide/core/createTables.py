import sqlite3 
import pandas as pd 

def anomalies():
    """Creates the anomalies table in the local SQLite DB if it doesn't exist."""
    try:
        with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='anomalies';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS anomalies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        server TEXT NOT NULL,
                        metric TEXT NOT NULL, -- 'CPU', 'MEM', 'DISK'
                        value REAL NOT NULL, -- anomalous_value 
                        severity REAL NOT NULL,
                        ai_summary TEXT,
                        ai_recommendation TEXT)
                """)
                conn.commit()
            else:
                pass
    except Exception as e:
        print(f'Error while creating the anomalies table: {e} ')


def retrainLog():
    """"Creates a retraining to show the progress and status of each servers model retraining """
    try:
        with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='retrainLog';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS retrainLog (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        modelName TEXT NOT NULL, -- name of the model being retrained
                        server TEXT NOT NULL,
                        status TEXT NOT NULL, -- 'successful', 'failed', 'rejected'
                        trainingLoss REAL,
                        lossConvergencePlot TEXT, -- path to the loss convergence plot image
                        message TEXT,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                    )
                """)
                conn.commit()
            else:
                pass
    except Exception as e:
        print(f'Error while creating the retrainLog table: {e} ')


def featureImportance(serverName):
    """creates a feature importance to view the feature causing the anomaly """
    try:
        with sqlite3.connect('ServerSide/database/featImportance.sqlite3') as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            c = conn.cursor()
            c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{serverName}';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS alertUsers (
                        server TEXT,
                        feature TEXT,
                        importance TEXT,
                        dateCreated TEXT
                    )
                """)

                conn.commit()
            else:
                pass
    except Exception as e:
        print(f'Error while creating the create_alert_users_table table: {e} ')



def alertUsers():
    """Registered users who receive alerts."""
    try:
        with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alertUsers';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS alertUsers (
                        username TEXT,
                        server TEXT,
                        ipAddress TEXT,
                        email TEXT,
                        dateCreated TEXT
                    )
                """)

                conn.commit()
            else:
                pass
    except Exception as e:
        print(f'Error while creating the create_alert_users_table table: {e} ')



def createRegisteredEmails():
    """Registered emails to recieve alerts. """
    try:
        with sqlite3.connect('ServerSide/database/main.sqlite3') as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regEmails';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS regEmails (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT NOT NULL,
                        dateCreated TEXT
                          )""")
                conn.commit()
            else:
                pass
    except Exception as e:
        print(f'Error while creating the email regsitration table: {e}')




if __name__ == '__main__':
    anomalies()
    retrainLog()
    alertUsers()
    createRegisteredEmails()
    # ! Dont add the featureImportance function, it will be added in the model training script
    print("All tables created successfully.")