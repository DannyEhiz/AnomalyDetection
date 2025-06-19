app/
├── main.py                  # Entry point
├── api/                     # Route definitions
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── servers.py       # /servers endpoints
│   │   ├── models.py        # /models endpoints
│   │   ├── anomalies.py     # /anomalies endpoints
├── core/                    # App config, settings, logging, schedulers
│   ├── __init__.py
│   ├── config.json            # Environment configs
│   ├── scheduler.py         # Auto-retrain logic (e.g. every Sunday)
│   ├── logger.py               # For logging errors and info
│   ├── createTables.py         # auto create all necessary tables from start of application
│   ├──
├── models/                  # Pydantic models (like Django forms/serializers)
│   ├── __init__.py
│   ├── LSTM Models
│   ├── Scalers
│   ├── 
├── services/                # Business logic (model training, inference)
│   ├── __init__.py
│   ├── modelling.py           # Retrain LSTM or IF
│   ├── detector.py          # Perform anomaly detection
│   ├── advancedAlerting.py --------# Alerting
├── database/                # DB connections, models
│   ├── __init__.py
│   ├── connection.py        # Connect to SQLite, Postgres, etc.
│   ├── crud.py              # Query/insert/update records
├── static/                  # (Optional) Model artifacts, logs
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_servers.py
│   ├── test_anomalies.py
├── requirements.txt
├── .env