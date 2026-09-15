"""
Database Layer (SQLite)
==========================

PURPOSE
-------
Provides the persistence layer described in the System Architecture
diagram: a SQLite database storing every incident, its predicted
category, and its risk score, so the Dashboard, Incident List, and
Incident Detail screens all read from a single consistent source.

WHY SQLITE (for Methodology/Technology chapter)
----------------------------------------------------
SQLite requires no separate server process, ships with Python's standard
library, and is well-suited to a single-user prototype of this scale.
The schema is designed so migrating to PostgreSQL later (for a
multi-user production version) would require minimal changes -- the
same table structure and queries would work, only the connection layer
would change.

FUNCTIONS
---------
init_db()              -- creates the incidents table if it doesn't exist
load_initial_data(df)  -- one-time load from incidents_with_risk_scores.csv
insert_incident(...)   -- adds a newly classified incident (from Upload screen)
get_all_incidents()    -- returns all incidents as a DataFrame (for List/Dashboard screens)
get_incident_by_id(id) -- returns a single incident's full record (for Detail screen)
update_status(id, status) -- updates an incident's Status field
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "risk_monitor.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    Incident_ID TEXT PRIMARY KEY,
    Date_Reported TEXT,
    Description TEXT,
    Category TEXT,
    Severity TEXT,
    Financial_Impact_GBP REAL,
    Likelihood_Score INTEGER,
    Impact_Score INTEGER,
    Risk_Score INTEGER,
    Risk_Tier TEXT,
    Status TEXT,
    Model_Confidence TEXT
);
"""


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def load_initial_data(csv_path):
    """One-time load of the pre-scored dataset into the database.
    Only runs if the table is currently empty, so re-running the app
    doesn't duplicate rows."""
    conn = get_connection()
    existing_count = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    if existing_count > 0:
        conn.close()
        return existing_count

    df = pd.read_csv(csv_path)
    if "Status" not in df.columns:
        df["Status"] = "Open"
    if "Model_Confidence" not in df.columns:
        df["Model_Confidence"] = ""

    cols = ["Incident_ID", "Date_Reported", "Description", "Category", "Severity",
            "Financial_Impact_GBP", "Likelihood_Score", "Impact_Score", "Risk_Score",
            "Risk_Tier", "Status", "Model_Confidence"]
    df[cols].to_sql("incidents", conn, if_exists="append", index=False)
    conn.close()
    return len(df)


def insert_incident(incident_id, date_reported, description, category, severity,
                     financial_impact, likelihood_score, impact_score, risk_score,
                     risk_tier, model_confidence, status="Pending Approval"):
    conn = get_connection()
    conn.execute(
        """INSERT INTO incidents
           (Incident_ID, Date_Reported, Description, Category, Severity,
            Financial_Impact_GBP, Likelihood_Score, Impact_Score, Risk_Score,
            Risk_Tier, Status, Model_Confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (incident_id, date_reported, description, category, severity,
         financial_impact, likelihood_score, impact_score, risk_score,
         risk_tier, status, model_confidence)
    )
    conn.commit()
    conn.close()


def get_all_incidents():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM incidents", conn)
    conn.close()
    return df


def get_incident_by_id(incident_id):
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM incidents WHERE Incident_ID = ?", conn, params=(incident_id,))
    conn.close()
    return df.iloc[0] if len(df) > 0 else None


def update_status(incident_id, new_status):
    conn = get_connection()
    conn.execute("UPDATE incidents SET Status = ? WHERE Incident_ID = ?", (new_status, incident_id))
    conn.commit()
    conn.close()


def delete_incident(incident_id):
    """Permanently removes an incident from the database. Used from the
    Incident Detail screen, guarded by an explicit confirmation step in
    the UI since this action cannot be undone."""
    conn = get_connection()
    conn.execute("DELETE FROM incidents WHERE Incident_ID = ?", (incident_id,))
    conn.commit()
    conn.close()


def get_next_incident_id():
    conn = get_connection()
    result = conn.execute(
        "SELECT Incident_ID FROM incidents ORDER BY Incident_ID DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if result is None:
        return "OPR-0001"
    last_num = int(result[0].split("-")[1])
    return f"OPR-{last_num + 1:04d}"