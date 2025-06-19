import sqlite3 
import pandas as pd 

def create_alert_users_table():
    """Creates the alertUsers table if it doesn't exist."""
    try:
        with sqlite3.connect('EdgeDB.db') as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alertUsers';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS alertUsers (
                        Username TEXT,
                        Active INT,
                        MgtZone TEXT,
                        Server_List TEXT,
                        IPAddress TEXT,
                        CPU_thresh TEXT,
                        MEM_thresh TEXT,
                        DISK_thresh TEXT,
                        Emails TEXT,
                        AlertType TEXT,
                        Alerting_AI TEXT,
                        dateCreated TEXT
                    )
                """)

                conn.commit()
            else:
                pass
    except Exception as e:
        print(f'Error while creating the create_alert_users_table table: {e} ')


def createOpenProblems():
    """ Creates the openProblems table in the local SQLite DB if it doesn't exist. """
    try:
        with sqlite3.connect('EdgeDB.db') as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='openProblems';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
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
            else:
                pass
    except Exception as e:
        print(f'Error while creating the openProblems table: {e} ')

def createRegisteredEmails():
    """ Creates the registeredEmails table in the local SQLite DB if it doesn't exist. """
    try:
        with sqlite3.connect('EdgeDB.db') as conn:
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


def inactiveServerTable():
    """Creates the inactiveServers table in the local SQLite DB if it doesn't exist."""
    try:
        with sqlite3.connect('EdgeDB.db') as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='inactiveServers';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS inactiveServers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT,
                        server TEXT NOT NULL,
                        ip_address TEXT,
                        mgt_zone TEXT,
                        dateCreated TEXT
                    )
                """)
                conn.commit()
    except Exception as e:
        print(f'Error while creating the inactiveServers table: {e}')

def lastTelemetryTime():
    try:
        with sqlite3.connect('EdgeDB.db') as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='latestTelemetry';")
            table_exists = c.fetchone()
            if not table_exists:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS latestTelemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        LogTimestamp TEXT,
                        server TEXT NOT NULL,
                        ip_address TEXT,
                        mgt_zone TEXT,
                        latestLog TEXT
                    )
                """)
                conn.commit()
    except Exception as e:
        print(f'Error while creating the inactiveServers table: {e}')






if __name__ == '__main__':
    createOpenProblems()
    createRegisteredEmails()
    create_alert_users_table()
    inactiveServerTable()
    lastTelemetryTime()