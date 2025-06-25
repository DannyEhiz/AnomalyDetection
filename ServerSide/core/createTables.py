import sqlite3 
import pandas as pd 
import random, string 


def generateAutoID(length=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def modelTrainHistory():
    try:
        with sqlite3.connect('ServerSide/database/mini.sqlite3') as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trainHistory';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS trainHistory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        server TEXT NOT NULL,
                        lastTrained TEXT NOt NULL)
                """)
                conn.commit()
            else:
                pass
    except Exception as e:
        print(f'Error while creating the trainHistory table: {e} ')

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
                        ID TEXT PRIMARY KEY,
                        METRIC TEXT NOT NULL,
                        SEVERITY TEXT NOT NULL,
                        SOURCE TEXT NOT NULL,
                        TIMESTAMP TEXT NOT NULL,
                        AI_SUMMARY TEXT)
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
        print(f'Error while creating the featureImportance table: {e} ')



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
        print(f'Error while creating the alertUsers table: {e} ')

def timeStampKeeper():
    """Keeps the last time stamp each server sent info to the db"""
    try:
        with sqlite3.connect('ServerSide/database/featImportance.sqlite3') as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='latestTime';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS latestTime (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        server TEXT,
                        lastTime TEXT,
                    )
                """)
                conn.commit()
            else:
                pass
    except Exception as e:
        print(f'Error while creating the latestTime table: {e} ')    



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
    timeStampKeeper()
    # ! Dont add the featureImportance function, it will be added in the model training script
    print("All tables created successfully.")