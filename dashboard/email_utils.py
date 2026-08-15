"""
Email Notification Utility
==============================

PURPOSE
-------
Sends a real email notification to a senior risk officer when a new
incident is submitted and requires approval (maker-checker control,
see app.py Submit Incident screen). This replaces the earlier
simulated-notification banner with an actual sent email.

CREDENTIALS -- KEPT PRIVATE
-------------------------------
Email credentials (EMAIL_ADDRESS, EMAIL_APP_PASSWORD) and the
recipient (APPROVER_EMAIL) are loaded from a local .env file via
python-dotenv, NEVER hardcoded in this script. The .env file must be
listed in .gitignore so it is never committed to GitHub -- see
.env.example for the required format (safe to commit, contains no
real values).

WHY GMAIL APP PASSWORDS (not your normal Gmail password)
--------------------------------------------------------------
Gmail blocks SMTP login using your regular account password for
security reasons. An "App Password" is a 16-character code generated
specifically for this purpose, which can be revoked independently of
your main password. Setup instructions are in the accompanying README
section / chat message.

WHY THIS IS WRAPPED IN TRY/EXCEPT
--------------------------------------
If credentials are missing or misconfigured, or the network/SMTP
server is unavailable, email sending should fail gracefully rather
than crashing the incident submission -- the incident is still saved
to the database with 'Pending Approval' status regardless of whether
the notification email succeeds.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
APPROVER_EMAIL = os.getenv("APPROVER_EMAIL", EMAIL_ADDRESS)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def send_approval_notification(incident_id, description, category, severity, risk_score, risk_tier):
    """Sends a real email to the configured approver notifying them a
    new incident needs review. Returns (success: bool, message: str)."""

    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        return False, (
            "Email not sent — EMAIL_ADDRESS / EMAIL_APP_PASSWORD not configured in .env. "
            "See .env.example for setup instructions."
        )

    subject = f"[Risk Monitor] Approval needed: {incident_id} ({risk_tier} risk)"
    body = f"""A new operational risk incident has been submitted and requires your approval.

Incident ID: {incident_id}
Category: {category}
Severity: {severity}
Risk Score: {risk_score} ({risk_tier} tier)

Description:
{description}

Please log in to the Risk Monitor dashboard to review and approve or reject this incident.

This is an automated notification from the AI-Driven Operational Risk Monitoring prototype.
"""

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = APPROVER_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            server.send_message(msg)
        return True, f"Notification email sent to {APPROVER_EMAIL}."
    except smtplib.SMTPAuthenticationError:
        return False, (
            "Email failed — authentication error. Check EMAIL_ADDRESS and EMAIL_APP_PASSWORD "
            "in .env (must be a Gmail App Password, not your regular password)."
        )
    except Exception as e:
        return False, f"Email failed to send: {e}"
