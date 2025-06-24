
# 🧠 Anomaly Detection Dashboard for Server Telemetry

A production-ready FastAPI + React-based anomaly detection platform for monitoring server infrastructure metrics (CPU, Memory, Disk, Network). This application uses AI models (LSTM Autoencoders, Isolation Forest) to detect unusual behaviors and alert stakeholders in real-time.

---

## 🚀 Features

- 📊 **Real-time anomaly detection** on server telemetry data
- 🧠 **AI-powered analysis** to explain detected anomalies
- 🔁 **Automatic weekly retraining** with newly ingested data
- 🔒 **Model validation enforcement** — rejects models with poor validation loss
- 📥 **Automated data download** from remote host every week
- 📧 **Email alerts** sent to registered users on anomaly detection
- 📅 **Scheduled retraining every Sunday** (customizable)
- 📈 **Interactive dashboard** showing model performance, anomaly timelines, and retrain logs

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, Pydantic, Celery, SQLite/PostgreSQL
- **ML Engine:** LSTM Autoencoder, Isolation Forest (Scikit-learn, TensorFlow/Keras)
- **Frontend:** React + TailwindCSS
- **Scheduler:** APScheduler / Celery Beat
- **Notifications:** SMTP email integration
- **Monitoring:** Matplotlib + Plotly for visual insights

---

## 🧱 Project Structure

```

app/
├── api/                # FastAPI route handlers
├── models/             # Pydantic schemas
├── services/           # Model training, anomaly detection, validation
├── database/           # SQLAlchemy models and CRUD logic
├── core/               # Configs, scheduler, email utils
├── main.py             # FastAPI entry point
frontend/
├── components/         # React components (charts, cards, modals)
├── pages/              # Server dashboard, anomaly views
├── App.tsx             # React entry point

````

---

## ⚙️ How It Works

1. **Each server** has its own anomaly detection model.
2. Models are trained weekly on **cleaned, validated telemetry data**.
3. If the new model's validation loss is **above threshold**, it is rejected.
4. **All anomalies** are automatically:
   - Flagged visually on the dashboard
   - Explained using AI text generation (LLM prompt)
   - Logged to the database
   - Sent via email to subscribed users
5. All data is automatically fetched from a configured host during retrain.

---

## 📅 Retraining Schedule

- 🕒 Models auto-retrain every **Sunday at 2:00 AM**
- ✅ Only accepted if validation loss is below configured threshold
- 🔁 Full logs and scores viewable on the dashboard

---

## 📧 Email Notifications

- Users can register emails for specific servers
- Alerts are sent when:
  - Anomaly is detected
  - Model retrain fails or is rejected
  - Retrain completes with summary

---

## 🧠 AI-Powered Anomaly Insight

Each anomaly is passed to an AI agent (e.g., LLM) that summarizes:
- What changed
- What metrics were involved
- Suggested reasons or interpretations

---

## 🧪 Development Setup

```bash
# Backend
cd app/
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend/
npm install
npm run dev
````


