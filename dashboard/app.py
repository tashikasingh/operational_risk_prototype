"""
AI-Driven Operational Risk Monitoring Dashboard
===================================================

Run from project root with:
    streamlit run dashboard/app.py

SCREENS (matches the wireframe document)
--------------------------------------------
1. Main Dashboard  -- KPIs, category/risk-tier/trend charts, top-5 highest risk
2. Incident List   -- filterable table of all incidents
3. Submit Incident -- free-text entry, live classification + risk scoring
4. Incident Detail -- full breakdown of one incident, incl. model confidence

SCOPE NOTE
------------
No login screen is implemented in this prototype (see wireframe doc
Section 2 -- explicitly stated as a scope limitation, not an oversight).
"""

import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import db
import email_utils
import insights_engine

st.set_page_config(page_title="Operational Risk Monitor", layout="wide", page_icon="\U0001F6E1\ufe0f")

# ---------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------
# Palette: deep navy "control room" sidebar (evokes a risk operations
# centre) against a clean light workspace, with a single consistent
# risk-tier colour code (green/amber/orange/red) used everywhere an
# incident's severity appears -- the signature element tying every
# screen together.
COLORS = {
    "bg": "#0A1120", "surface": "#20304D", "navy": "#060B15", "navy_light": "#2C3F60",
    "accent": "#5B8FD6", "accent_bright": "#8FC4FF", "text": "#F0F3F8", "muted": "#9AA7BC", "border": "#3A4D6E",
    "low": "#3DB56E", "medium": "#E3A83B", "high": "#E2793A", "critical": "#D64545",
}

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {COLORS['text']}; }}
    .stApp {{ background-color: {COLORS['bg']}; }}
    h1, h2, h3 {{ font-family: 'Sora', sans-serif !important; color: {COLORS['text']} !important; }}
    p, span, label, .stMarkdown, .stCaption {{ color: {COLORS['text']}; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{ background-color: {COLORS['navy']}; }}
    section[data-testid="stSidebar"] * {{ color: #E8EAF0 !important; }}
    section[data-testid="stSidebar"] h1 {{
        font-family: 'Sora', sans-serif !important; font-size: 1.3rem !important;
        color: #FFFFFF !important; border-bottom: 1px solid {COLORS['border']}; padding-bottom: 12px;
    }}
    section[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
        background-color: transparent; border-radius: 8px; padding: 8px 12px; margin-bottom: 4px;
        transition: background-color 0.15s ease;
    }}
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
        background-color: {COLORS['navy_light']};
    }}

    /* Eyebrow label above page titles */
    .eyebrow {{
        font-family: 'Inter', sans-serif; font-size: 0.85rem; letter-spacing: 0.05em;
        color: #FFFFFF; text-transform: uppercase; margin-bottom: -4px; font-weight: 700;
    }}

    /* KPI cards -- icon now sits in its own colour-tinted badge circle for a cleaner, more corporate look */
    .kpi-row {{ display: flex; gap: 16px; margin: 8px 0 20px 0; flex-wrap: wrap; }}
    .kpi-card {{
        background: {COLORS['surface']}; border-radius: 12px; padding: 16px 20px; flex: 1; min-width: 190px;
        border: 1px solid {COLORS['border']}; border-left: 4px solid var(--accent-color);
        display: flex; flex-direction: column; gap: 8px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }}
    .kpi-icon-badge {{
        width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
        font-size: 1.05rem; background-color: color-mix(in srgb, var(--accent-color) 18%, transparent);
    }}
    .kpi-label {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: {COLORS['muted']};
        text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .kpi-value {{ font-family: 'Sora', sans-serif; font-size: 1.85rem; font-weight: 700; color: {COLORS['text']}; }}

    /* Chart / content cards -- every chart now sits inside a bordered card matching the KPI cards,
       so the page reads as a consistent grid rather than headers floating on bare background */
    .chart-card {{
        background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 12px;
        padding: 18px 20px 8px 20px; margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }}
    .chart-card-title {{
        font-family: 'Sora', sans-serif; font-size: 1.05rem; font-weight: 700; color: {COLORS['text']};
        margin-bottom: 4px;
    }}

    /* Risk tier badges -- the signature element, used everywhere */
    .badge {{
        display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.76rem;
        font-weight: 700; font-family: 'Inter', sans-serif; color: #0F1B2E;
    }}
    .badge-low {{ background-color: {COLORS['low']}; color: #06210F; }}
    .badge-medium {{ background-color: {COLORS['medium']}; color: #2B1D02; }}
    .badge-high {{ background-color: {COLORS['high']}; color: #2B1502; }}
    .badge-critical {{ background-color: {COLORS['critical']}; color: #2B0505; }}

    /* Custom HTML tables -- long text truncates with an ellipsis + native tooltip on hover,
       so every row stays a uniform height instead of wrapping to 2-3 lines */
    .risk-table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; background: {COLORS['surface']};
        border-radius: 10px; overflow: hidden; border: 1px solid {COLORS['border']}; }}
    .risk-table th {{
        background-color: {COLORS['navy']}; color: #E8EAF0 !important; text-align: left; padding: 10px 12px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .risk-table td {{
        padding: 9px 12px; border-bottom: 1px solid {COLORS['border']}; color: {COLORS['text']};
        max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    .risk-table tr:last-child td {{ border-bottom: none; }}
    .risk-table tr:hover td {{ background-color: {COLORS['navy_light']}; }}
    .incident-id {{ font-family: 'Inter', sans-serif; color: #FFFFFF; font-weight: 700; font-size: 0.95rem; }}

    /* Cards for cluster/theme lists etc */
    .theme-card {{
        background: {COLORS['surface']}; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
        border: 1px solid {COLORS['border']}; border-left: 4px solid {COLORS['accent']};
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['accent']}; color: white; border-radius: 8px; border: none;
        font-family: 'Inter', sans-serif; font-weight: 600;
    }}
    .stButton > button:hover {{ background-color: {COLORS['navy_light']}; color: white; }}

    /* Native Streamlit input widgets -- overridden so they match the dark surface instead of
       defaulting to light styling that clashes with the rest of the page */
    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {{
        background-color: {COLORS['surface']} !important; color: {COLORS['text']} !important;
        border-color: {COLORS['border']} !important;
    }}
    div[data-testid="stExpander"] {{
        background-color: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 8px;
    }}
    div[data-testid="stExpander"] summary {{ color: {COLORS['text']} !important; }}

    /* Streamlit's native info/warning/success/error boxes -- overridden so
       text stays readable against the dark theme instead of defaulting to
       light backgrounds with low-contrast text */
    div[data-testid="stAlertContainer"] {{
        background-color: {COLORS['navy_light']} !important; border: 1px solid {COLORS['border']};
    }}
    div[data-testid="stAlertContainer"] p, div[data-testid="stAlertContainer"] span,
    div[data-testid="stAlertContainer"] div {{ color: {COLORS['text']} !important; }}
    div[data-testid="stMetricValue"], div[data-testid="stMarkdownContainer"] {{ color: {COLORS['text']}; }}
    hr {{ border-color: {COLORS['border']}; }}
    </style>
    """, unsafe_allow_html=True)

BADGE_CLASS = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high", "Critical": "badge-critical"}

# Consistent colour per Basel category, used for both the clustering chart
# and the recommendation cards on the Insights page, so a category means
# the same colour everywhere in the app.
CATEGORY_COLORS = {
    "External Fraud": "#5B8FD6",
    "Business Disruption & System Failures": "#6FD1E8",
    "Cybersecurity Incident": "#D64545",
    "Employment Practices & Workplace Safety": "#E38BA8",
    "Internal Fraud": "#3DB56E",
    "Clients, Products & Business Practices": "#8FE0A0",
    "Damage to Physical Assets": "#E3A83B",
    "Execution, Delivery & Process Management": "#B08FE0",
}

def risk_badge(tier):
    cls = BADGE_CLASS.get(tier, "badge-medium")
    return f'<span class="badge {cls}">{tier}</span>'

def kpi_card(icon, label, value, accent):
    return f"""<div class="kpi-card" style="--accent-color:{accent}">
        <div class="kpi-icon-badge" style="color:{accent}">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>"""

def render_kpis(cards):
    html = '<div class="kpi-row">' + "".join(cards) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

def chart_card_open(title):
    st.markdown(f'<div class="chart-card"><div class="chart-card-title">{title}</div>', unsafe_allow_html=True)

def chart_card_close():
    st.markdown('</div>', unsafe_allow_html=True)

def render_html_table(df, badge_col=None, id_col="Incident_ID", truncate_col=None, truncate_len=55, tooltip_cols=None):
    """Renders a DataFrame as a styled HTML table with risk-tier badges,
    avoiding st.dataframe/st.table entirely (side-steps pyarrow dependency
    issues and gives full control over the visual design). truncate_col gets
    hard-sliced with an ellipsis; tooltip_cols get a title="" attribute so
    the full value shows as a native tooltip on hover, relying on the CSS
    max-width/ellipsis rule to keep the cell itself visually short."""
    df = df.copy()
    for col in (tooltip_cols or []):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f'<span title="{x}">{x}</span>')
    if truncate_col and truncate_col in df.columns:
        df[truncate_col] = df[truncate_col].apply(
            lambda x: f'<span title="{x}">{x[:truncate_len]}{"..." if len(str(x)) > truncate_len else ""}</span>'
        )
    if badge_col and badge_col in df.columns:
        df[badge_col] = df[badge_col].apply(risk_badge)
    if id_col and id_col in df.columns:
        df[id_col] = df[id_col].apply(lambda x: f'<span class="incident-id">{x}</span>')
    html = df.to_html(escape=False, index=False, classes="risk-table", border=0)
    st.markdown(html, unsafe_allow_html=True)

def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)

inject_css()

# ---------------------------------------------------------------------
# Load model, vectoriser, and scoring lookup once per session
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/naive_bayes_model.pkl")
    with open("models/risk_scoring_lookup.json") as f:
        lookup = json.load(f)
    return vectorizer, model, lookup

vectorizer, model, lookup = load_model_artifacts()
category_likelihood = lookup["category_likelihood"]
severity_to_impact = lookup["severity_to_impact"]

# ---------------------------------------------------------------------
# Initialise database (one-time)
# ---------------------------------------------------------------------
db.init_db()
db.load_initial_data("data/processed/incidents_with_risk_scores.csv")


def risk_tier(score):
    if score <= 5:
        return "Low"
    elif score <= 10:
        return "Medium"
    elif score <= 15:
        return "High"
    else:
        return "Critical"


def classify_and_score(description, severity):
    """Runs the same TF-IDF -> Naive Bayes -> risk scoring pipeline
    used in train_classifier.py / risk_scoring.py, applied live to a
    single new incident description."""
    desc_clean = description.strip().lower()
    X = vectorizer.transform([desc_clean])
    predicted_category = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    class_probs = dict(zip(model.classes_, probs))
    top_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:4]

    likelihood_score = category_likelihood.get(predicted_category, 3)
    impact_score = severity_to_impact.get(severity, 2)
    risk_score = likelihood_score * impact_score
    tier = risk_tier(risk_score)

    confidence_str = "; ".join([f"{c}: {p*100:.1f}%" for c, p in top_probs])
    return predicted_category, likelihood_score, impact_score, risk_score, tier, confidence_str, top_probs


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.markdown("# \U0001F6E1\ufe0f Risk Monitor")
page = st.sidebar.radio("Navigate", [
    "\U0001F4CA  Main Dashboard", "\U0001F4CB  Incident List", "\U0001F4DD  Submit Incident",
    "\U0001F50D  Incident Detail", "\U0001F4A1  Insights", "\u2753  Help"
], label_visibility="collapsed")
page = page.split("  ", 1)[1]  # strip icon for logic below

if "selected_incident_id" not in st.session_state:
    st.session_state.selected_incident_id = None

# =======================================================================
# SCREEN 1: MAIN DASHBOARD
# =======================================================================
if page == "Main Dashboard":
    eyebrow("OVERVIEW")
    st.title("Operational Risk Monitor")

    df = db.get_all_incidents()

    n_open = len(df[df["Status"].isin(["Open", "Under Investigation"])])
    n_pending = len(df[df["Status"] == "Pending Approval"])
    render_kpis([
        kpi_card("\U0001F4C1", "Total Incidents", len(df), COLORS["accent"]),
        kpi_card("\u26A0\ufe0f", "High / Critical Risk", len(df[df["Risk_Tier"].isin(["High", "Critical"])]), COLORS["high"]),
        kpi_card("\U0001F4C8", "Average Risk Score", f"{df['Risk_Score'].mean():.1f}", COLORS["medium"]),
        kpi_card("\U0001F550", "Open / Under Investigation", n_open, COLORS["navy"]),
        kpi_card("\U0001F4E9", "Pending Approval", n_pending, COLORS["medium"]),
    ])

    col_a, col_b = st.columns([3, 2])
    with col_a:
        chart_card_open("Incidents by Category")
        cat_stats = df.groupby("Category").agg(
            Count=("Incident_ID", "count"),
            Avg_Risk_Score=("Risk_Score", "mean")
        ).reset_index()
        cat_stats["Pct"] = (cat_stats["Count"] / len(df) * 100).round(1)
        cat_stats = cat_stats.sort_values("Count", ascending=True)  # ascending so largest bar is on top
        # Horizontal orientation avoids the rotated/overlapping label problem
        # that long Basel category names cause on a vertical bar chart.
        fig = px.bar(cat_stats, x="Count", y="Category", orientation="h",
                     custom_data=["Pct", "Avg_Risk_Score"])
        fig.update_traces(
            marker_color=COLORS["accent_bright"],
            hovertemplate="<b>%{y}</b><br>Incidents: %{x} (%{customdata[0]}% of total)<br>"
                          "Avg Risk Score: %{customdata[1]:.1f}<extra></extra>"
        )
        fig.update_layout(
            height=480, plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"],
            font_color=COLORS["text"], xaxis_gridcolor=COLORS["border"], yaxis_gridcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10), yaxis_title="", xaxis_title="Incidents",
            yaxis=dict(tickfont=dict(size=13)),
        )
        st.plotly_chart(fig, width='stretch')
        chart_card_close()

    with col_b:
        chart_card_open("Risk Tier Breakdown")
        tier_stats = df.groupby("Risk_Tier").agg(
            Count=("Incident_ID", "count"),
            Avg_Financial_Impact=("Financial_Impact_GBP", "mean")
        ).reset_index()
        fig = px.pie(tier_stats, names="Risk_Tier", values="Count",
                     color="Risk_Tier", custom_data=["Avg_Financial_Impact"], hole=0.45,
                     color_discrete_map={"Low": COLORS["low"], "Medium": COLORS["medium"],
                                         "High": COLORS["high"], "Critical": COLORS["critical"]})
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Incidents: %{value} (%{percent})<br>"
                          "Avg Financial Impact: £%{customdata[0]:,.0f}<extra></extra>",
            textfont_color="#0F1B2E"
        )
        fig.update_layout(
            height=480, paper_bgcolor=COLORS["surface"], font_color=COLORS["text"],
            legend=dict(font=dict(color=COLORS["text"])), margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width='stretch')
        chart_card_close()

    chart_card_open("Risk Trend Over Time")
    df["Date_Reported"] = pd.to_datetime(df["Date_Reported"], errors="coerce")
    trend = df.dropna(subset=["Date_Reported"]).copy()
    trend["Month"] = trend["Date_Reported"].dt.to_period("M").astype(str)
    monthly = trend.groupby("Month").agg(
        Risk_Score=("Risk_Score", "mean"),
        Incident_Count=("Incident_ID", "count")
    ).reset_index()
    fig = px.line(monthly, x="Month", y="Risk_Score", markers=True,
                  custom_data=["Incident_Count"])
    fig.update_traces(
        line_color=COLORS["accent"], marker_color=COLORS["accent"], marker_size=6,
        hovertemplate="<b>%{x}</b><br>Avg Risk Score: %{y:.1f}<br>"
                      "Incidents that month: %{customdata[0]}<extra></extra>"
    )
    fig.update_layout(
        height=330, yaxis_title="Average Risk Score", plot_bgcolor=COLORS["surface"],
        paper_bgcolor=COLORS["surface"], font_color=COLORS["text"],
        xaxis_gridcolor=COLORS["border"], yaxis_gridcolor=COLORS["border"],
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, width='stretch')
    chart_card_close()

    chart_card_open("Top 5 Highest Risk Incidents")
    st.caption(
        "Ranked by Risk Score, descending. Ties are broken by Financial Impact (higher first), "
        "then by most recently reported — reflecting that between two incidents of equal Likelihood × "
        "Impact score, the one with greater actual financial exposure and more recent occurrence "
        "warrants earlier attention."
    )
    top5 = df.sort_values(
        ["Risk_Score", "Financial_Impact_GBP", "Date_Reported"],
        ascending=[False, False, False]
    ).head(5)[["Incident_ID", "Description", "Category", "Financial_Impact_GBP", "Risk_Score", "Risk_Tier"]].copy()
    top5.insert(0, "Rank", [f"#{i+1}" for i in range(len(top5))])
    top5["Financial_Impact_GBP"] = top5["Financial_Impact_GBP"].apply(lambda x: f"£{x:,.0f}")
    render_html_table(top5, badge_col="Risk_Tier", truncate_col="Description", truncate_len=70)
    chart_card_close()

# =======================================================================
# SCREEN 2: INCIDENT LIST
# =======================================================================
elif page == "Incident List":
    eyebrow("BROWSE & FILTER")
    st.title("Incident List")

    df = db.get_all_incidents()

    f1, f2, f3 = st.columns(3)
    with f1:
        cat_filter = st.multiselect("Category", sorted(df["Category"].unique()))
    with f2:
        tier_filter = st.multiselect("Risk Tier", ["Low", "Medium", "High", "Critical"])
    with f3:
        search = st.text_input("Search description")

    filtered = df.copy()
    if cat_filter:
        filtered = filtered[filtered["Category"].isin(cat_filter)]
    if tier_filter:
        filtered = filtered[filtered["Risk_Tier"].isin(tier_filter)]
    if search:
        filtered = filtered[filtered["Description"].str.contains(search, case=False, na=False)]

    st.write(f"Showing {len(filtered)} of {len(df)} incidents")
    if len(filtered) > 50:
        st.caption("Displaying first 50 rows — use the filters above to narrow results further.")

    display_df = filtered[["Incident_ID", "Date_Reported", "Description", "Category",
                            "Severity", "Risk_Tier", "Status"]].copy()

    render_html_table(display_df.head(50).reset_index(drop=True), badge_col="Risk_Tier",
                      truncate_col="Description", truncate_len=65, tooltip_cols=["Category"])

    selected_id = st.selectbox("Open incident detail for:", [""] + filtered["Incident_ID"].tolist())
    if selected_id:
        st.session_state.selected_incident_id = selected_id
        st.info(f"Selected {selected_id} — go to 'Incident Detail' in the sidebar to view it.")

# =======================================================================
# SCREEN 3: SUBMIT INCIDENT
# =======================================================================
elif page == "Submit Incident":
    eyebrow("NEW ENTRY")
    st.title("Submit New Incident")

    description = st.text_area("Incident Description", height=150,
                                placeholder="Describe what happened...")
    col1, col2 = st.columns(2)
    with col1:
        severity = st.selectbox("Severity (analyst estimate)", ["Low", "Medium", "High", "Critical"])
    with col2:
        financial_impact = st.number_input("Estimated Financial Impact (GBP)", min_value=0, value=0)

    if st.button("Classify & Submit", type="primary"):
        if not description.strip():
            st.error("Please enter a description first.")
        else:
            predicted_category, likelihood, impact, risk_score, tier, confidence_str, top_probs = \
                classify_and_score(description, severity)

            st.session_state.pending_incident = {
                "description": description, "severity": severity,
                "financial_impact": financial_impact, "predicted_category": predicted_category,
                "likelihood": likelihood, "impact": impact, "risk_score": risk_score,
                "tier": tier, "confidence_str": confidence_str, "top_probs": top_probs,
            }

    if "pending_incident" in st.session_state:
        p = st.session_state.pending_incident
        st.divider()
        st.subheader("Prediction Result")
        st.write(f"**Predicted Category:** {p['predicted_category']}")
        st.write(f"**Risk Score:** {p['risk_score']} ({p['tier']} — Likelihood {p['likelihood']} × Impact {p['impact']})")

        probs_df = pd.DataFrame(p["top_probs"], columns=["Category", "Probability"])
        fig = px.bar(probs_df, x="Probability", y="Category", orientation="h")
        fig.update_traces(marker_color=COLORS["accent"])
        fig.update_layout(height=250, xaxis_tickformat=".0%", plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font_color=COLORS["text"], xaxis_gridcolor=COLORS["border"], yaxis_gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            final_category = st.selectbox("Confirm or override category",
                                            [p["predicted_category"]] +
                                            [c for c in category_likelihood if c != p["predicted_category"]])
        with c2:
            button_label = "Confirm & Save" if p["severity"] in ("Low", "Medium") else "Confirm & Send for Approval"
            if st.button(button_label):
                # Severity-based approval threshold: per practitioner interview
                # feedback (Participant A, AML risk analyst), only High/Critical
                # severity incidents require senior review before finalisation;
                # lower-severity incidents can be logged directly as Open. This
                # mirrors real institutional practice more closely than requiring
                # approval for every incident regardless of severity.
                requires_approval = p["severity"] in ("High", "Critical")
                initial_status = "Pending Approval" if requires_approval else "Open"

                new_id = db.get_next_incident_id()
                db.insert_incident(
                    incident_id=new_id,
                    date_reported=datetime.now().strftime("%Y-%m-%d"),
                    description=p["description"],
                    category=final_category,
                    severity=p["severity"],
                    financial_impact=p["financial_impact"],
                    likelihood_score=p["likelihood"],
                    impact_score=p["impact"],
                    risk_score=p["risk_score"],
                    risk_tier=p["tier"],
                    model_confidence=p["confidence_str"],
                    status=initial_status,
                )
                st.success(f"Saved as {new_id} — status: **{initial_status}**")

                if requires_approval:
                    with st.spinner("Sending approval notification email..."):
                        email_sent, email_message = email_utils.send_approval_notification(
                            incident_id=new_id,
                            description=p["description"],
                            category=final_category,
                            severity=p["severity"],
                            risk_score=p["risk_score"],
                            risk_tier=p["tier"],
                        )
                    if email_sent:
                        st.info(f"\U0001F4E7 {email_message} A maker-checker control: {p['severity']}-severity "
                                "incidents stay Pending Approval until a senior risk officer approves or "
                                "rejects them from the Incident Detail screen — this threshold reflects "
                                "direct practitioner feedback that only higher-severity incidents warrant "
                                "senior review before finalisation.")
                    else:
                        st.warning(f"\u26A0\ufe0f {email_message} The incident was still saved and remains "
                                   "Pending Approval — only the email notification failed.")
                else:
                    st.info(
                        f"\u2139\ufe0f {p['severity']}-severity incidents do not require senior approval "
                        "before finalisation (per practitioner feedback that only High/Critical incidents "
                        "warrant that review step), so this incident is already Open."
                    )
                del st.session_state.pending_incident

# =======================================================================
# SCREEN 4: INCIDENT DETAIL
# =======================================================================
elif page == "Incident Detail":
    eyebrow("CASE FILE")
    st.title("Incident Detail")

    df = db.get_all_incidents()
    default_id = st.session_state.selected_incident_id
    incident_id = st.selectbox(
        "Select Incident ID",
        df["Incident_ID"].tolist(),
        index=df["Incident_ID"].tolist().index(default_id) if default_id in df["Incident_ID"].tolist() else 0
    )

    record = db.get_incident_by_id(incident_id)
    if record is not None:
        st.subheader(f"{record['Incident_ID']}")
        st.write(f"**Description:** {record['Description']}")

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Predicted Category:** {record['Category']}")
            st.write(f"**Severity:** {record['Severity']}")
            st.write(f"**Financial Impact:** £{record['Financial_Impact_GBP']:,.0f}")

            if record["Model_Confidence"]:
                st.write("**Model Confidence:**")
                pairs = [p.split(": ") for p in record["Model_Confidence"].split("; ") if ": " in p]
                if pairs:
                    conf_df = pd.DataFrame(pairs, columns=["Category", "Probability"])
                    conf_df["Probability"] = conf_df["Probability"].str.rstrip("%").astype(float)
                    fig = px.bar(conf_df, x="Probability", y="Category", orientation="h")
                    fig.update_traces(marker_color=COLORS["accent"])
                    fig.update_layout(height=250, plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font_color=COLORS["text"], xaxis_gridcolor=COLORS["border"], yaxis_gridcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, width='stretch')

        with c2:
            st.write(f"**Likelihood Score:** {record['Likelihood_Score']} / 5")
            st.write(f"**Impact Score:** {record['Impact_Score']} / 5")
            st.write(f"**Risk Score:** {record['Likelihood_Score']} × {record['Impact_Score']} = {record['Risk_Score']}")
            st.write(f"**Risk Tier:**")
            st.markdown(risk_badge(record['Risk_Tier']), unsafe_allow_html=True)

            st.write("**Status:**")
            if record["Status"] == "Pending Approval":
                st.warning("\u23F3 Awaiting senior risk officer approval before this incident is finalised.")
                ac1, ac2 = st.columns(2)
                with ac1:
                    if st.button("\u2705 Approve", type="primary"):
                        db.update_status(incident_id, "Open")
                        st.success("Approved — status set to Open.")
                        st.rerun()
                with ac2:
                    if st.button("\u274C Reject"):
                        db.update_status(incident_id, "Rejected")
                        st.error("Rejected — incident flagged for review/correction.")
                        st.rerun()
            else:
                status_options = ["Pending Approval", "Open", "Under Investigation", "Escalated", "Resolved", "Closed", "Rejected"]
                new_status = st.selectbox(
                    "Change status", status_options,
                    index=status_options.index(record["Status"]) if record["Status"] in status_options else 0
                )
                if st.button("Update Status"):
                    db.update_status(incident_id, new_status)
                    st.success("Status updated.")

        st.divider()
        with st.expander("\U0001F5D1\ufe0f Delete this incident"):
            st.warning(
                "This permanently removes the incident from the database. This cannot be undone, "
                "and it will not be reflected in Insights until you next click Refresh Insights."
            )
            confirm_delete = st.checkbox(f"I understand this will permanently delete {incident_id}")
            if st.button("Delete Incident", disabled=not confirm_delete, type="primary"):
                db.delete_incident(incident_id)
                st.session_state.selected_incident_id = None
                st.success(f"{incident_id} deleted.")
                st.rerun()

# =======================================================================
# SCREEN 5: INSIGHTS (root-cause analysis, trends, anomalies)
# =======================================================================
elif page == "Insights":
    eyebrow("ROOT-CAUSE ANALYSIS")
    st.title("Insights")
    st.caption("Why incidents happen, and what to do about it.")
    st.caption("Goes beyond classification to surface root causes, patterns, trends, and anomalies.")

    # -----------------------------------------------------------------
    # Refresh control -- these analyses are cached by default (fast to
    # load), but were computed from a static snapshot and do NOT
    # automatically include incidents submitted/approved since then.
    # Clicking Refresh recomputes all four analyses live against the
    # current database contents.
    # -----------------------------------------------------------------
    rc1, rc2 = st.columns([3, 1])
    with rc1:
        st.info(
            "\u2139\ufe0f These insights are cached and may not reflect incidents submitted since the "
            "last refresh. Click **Refresh Insights** to recompute from the current database "
            "(may take a few seconds)."
        )
    with rc2:
        refresh_clicked = st.button("\U0001F504 Refresh Insights", type="primary")

    if refresh_clicked:
        with st.spinner("Recomputing clustering, association rules, trend forecast, and anomaly detection..."):
            live_df = db.get_all_incidents()
            results = insights_engine.run_all_insights(live_df)

            insights_df = results["incidents_with_insights"]
            clusters = results["cluster_summary"]
            rules_df = results["association_rules"]
            trend = results["trend_forecast"]

            # Persist so the next page load starts from these fresh results too
            insights_df.to_csv("data/processed/incidents_with_insights.csv", index=False)
            with open("data/processed/cluster_summary.json", "w") as f:
                json.dump(clusters, f, indent=2)
            rules_df.to_csv("data/processed/association_rules.csv", index=False)
            with open("data/processed/trend_forecast.json", "w") as f:
                json.dump(trend, f, indent=2)

        st.success(f"Insights refreshed — now based on all {len(insights_df)} current incidents.")
    else:
        insights_df = pd.read_csv("data/processed/incidents_with_insights.csv")
        with open("data/processed/cluster_summary.json") as f:
            clusters = json.load(f)
        rules_df = pd.read_csv("data/processed/association_rules.csv")
        with open("data/processed/trend_forecast.json") as f:
            trend = json.load(f)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Root-Cause Themes", "Association Patterns", "Trend Forecast", "Anomaly Detection"
    ])

    # -------------------------------------------------------------
    with tab1:
        st.subheader("Root-Cause Clusters (sub-themes within categories)")
        st.write("Incidents grouped by K-Means clustering on description text — reveals finer-grained "
                 "root causes than the 8 Basel categories alone.")

        clusters_sorted = sorted(clusters, key=lambda c: -c["size"])
        cluster_df = pd.DataFrame([{"Theme": c["label"], "Count": c["size"],
                                     "Dominant Category": c["dominant_category"]} for c in clusters_sorted])
        fig = px.bar(cluster_df, x="Count", y="Theme", orientation="h", color="Dominant Category",
                     color_discrete_map=CATEGORY_COLORS)
        fig.update_layout(
            height=450, yaxis={"categoryorder": "total ascending"},
            plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font_color=COLORS["text"],
            xaxis_gridcolor=COLORS["border"], yaxis_gridcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color=COLORS["text"])),
        )
        st.plotly_chart(fig, width='stretch')

        st.subheader("Recommended Actions per Root-Cause Theme")
        for c in clusters_sorted:
            cat_color = CATEGORY_COLORS.get(c["dominant_category"], COLORS["accent"])
            st.markdown(f"""
            <div class="theme-card" style="border-left-color: {cat_color};">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                    <span style="background-color:{cat_color}; color:#0F1B2E; font-weight:700;
                                 font-size:0.72rem; padding:2px 9px; border-radius:999px;">
                        {c['dominant_category']}
                    </span>
                    <span style="color:{COLORS['muted']}; font-size:0.8rem;">{c['size']} incidents</span>
                </div>
                <div style="font-weight:700; color:{COLORS['text']}; font-size:1rem; margin-bottom:6px;">
                    {c['label']}
                </div>
                <div style="color:{COLORS['text']}; font-size:0.9rem; line-height:1.4;">
                    \u2192 {c['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    with tab2:
        st.subheader("Association Patterns (Category, Severity, Department)")
        st.info("Note: Detection_Method was deliberately excluded from this analysis — it is "
                "near-deterministic per category in this synthetic dataset (a generation artifact), "
                "not a genuine real-world pattern. Only genuinely independent fields are shown below.")
        if len(rules_df) > 0:
            render_html_table(rules_df.reset_index(drop=True), id_col=None)
            st.caption(f"{len(rules_df)} rules found with lift > 1.3 and confidence > 0.4. "
                       "Modest lift values here reflect that Category/Severity/Department were "
                       "not deliberately correlated during dataset generation.")
        else:
            st.write("No rules met the support/confidence/lift thresholds.")

    # -------------------------------------------------------------
    with tab3:
        st.subheader("Incident Volume & Risk Trend")

        hist = pd.DataFrame(trend["historical"])
        fig = px.line(hist, x="Month", y="Incident_Count", markers=True, title="Monthly Incident Count")
        fig.update_traces(line_color=COLORS["accent"], marker_color=COLORS["accent"])
        fig.update_layout(height=350, plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font_color=COLORS["text"], xaxis_gridcolor=COLORS["border"], yaxis_gridcolor=COLORS["border"])
        st.plotly_chart(fig, width='stretch')

        if trend.get("excluded_current_month"):
            st.caption(
                "\u2139\ufe0f The current (still in-progress) month is shown on the chart above but is "
                "excluded from the significance test and forecast below — a partial month always looks "
                "artificially low simply because it hasn't finished yet, which would otherwise distort "
                "the trend calculation."
            )

        sig_count = trend["incident_count_trend_significant"]
        st.write(f"**Trend significance:** p = {trend['incident_count_p_value']:.3f} — "
                 f"{'a statistically significant trend was found' if sig_count else 'no statistically significant trend detected (p > 0.05)'}")
        if not sig_count:
            st.caption("This is expected: incident dates were randomly assigned during synthetic data "
                       "generation, so no genuine temporal trend exists. A real bank's data would likely "
                       "show genuine seasonal/business-driven trends this synthetic dataset doesn't model.")

        # Build readable month labels for the 3 forecast months (continuing
        # on from the last historical month) instead of showing raw index
        # numbers, and present as a table instead of a raw Python list.
        import pandas as _pd
        last_month_str = trend["historical"][-1]["Month"] if trend.get("historical") else None
        forecast_rows = []
        if last_month_str:
            last_period = _pd.Period(last_month_str, freq="M")
            for i in range(len(trend["forecast_incident_count"])):
                month_label = str(last_period + (i + 1))
                forecast_rows.append({
                    "Month": month_label,
                    "Forecast Incident Count": round(trend["forecast_incident_count"][i], 1),
                    "Forecast Avg Risk Score": round(trend["forecast_avg_risk_score"][i], 1),
                })

        if forecast_rows:
            st.write("**Next 3 Months Forecast**")
            forecast_df = pd.DataFrame(forecast_rows)
            render_html_table(forecast_df, id_col=None)

        # -------------------------------------------------------------
        # BACKTEST: validates the forecast against real held-out data,
        # rather than asking the viewer to trust the forecast blindly.
        # -------------------------------------------------------------
        backtest = trend.get("backtest", {})
        if backtest.get("available"):
            st.divider()
            chart_card_open("Model Validation: Predicted vs Actual (Backtest)")
            st.caption(
                "The last 3 complete months were held out and NOT shown to either model during fitting. "
                "Each method then predicted those months \u2014 compared here against what actually happened, "
                "so the forecast's accuracy can be judged directly rather than taken on trust."
            )

            bt_df = pd.DataFrame({
                "Month": backtest["months"],
                "Actual": backtest["actual"],
                "Holt's Forecast": [round(v, 1) for v in backtest["holt_prediction"]],
                "Moving Average": [round(v, 1) for v in backtest["moving_avg_prediction"]],
            })
            bt_melted = bt_df.melt(id_vars="Month", var_name="Series", value_name="Incident Count")

            fig = px.line(bt_melted, x="Month", y="Incident Count", color="Series", markers=True,
                          color_discrete_map={
                              "Actual": COLORS["text"],
                              "Holt's Forecast": COLORS["accent_bright"],
                              "Moving Average": COLORS["medium"],
                          })
            fig.update_traces(line=dict(width=3))
            for trace in fig.data:
                if trace.name == "Actual":
                    trace.line.width = 4
                    trace.line.dash = "solid"
                elif trace.name == "Holt's Forecast":
                    trace.line.dash = "solid"
                    trace.marker.symbol = "circle"
                elif trace.name == "Moving Average":
                    trace.line.dash = "dash"
                    trace.marker.symbol = "diamond"
            fig.update_layout(
                height=330, plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"],
                font_color=COLORS["text"], xaxis_gridcolor=COLORS["border"], yaxis_gridcolor=COLORS["border"],
                legend=dict(font=dict(color=COLORS["text"])), margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(type="category"),
            )
            st.plotly_chart(fig, width='stretch')

            bc1, bc2 = st.columns(2)
            with bc1:
                st.metric("Holt's Forecast — Mean Error", f"{backtest['holt_mae']:.2f} incidents/month")
            with bc2:
                st.metric("Moving Average — Mean Error", f"{backtest['moving_avg_mae']:.2f} incidents/month")

            diff = backtest["moving_avg_mae"] - backtest["holt_mae"]
            if abs(diff) < 0.5:
                st.info(
                    "\u2139\ufe0f Holt's Exponential Smoothing performs almost identically to a simple "
                    "moving average here. This is consistent with the earlier finding that no statistically "
                    "significant trend exists in this data (Section: Trend significance above) — when there "
                    "is no real trend to capture, a more sophisticated method offers little to no advantage "
                    "over a naive average. This is an honest result, not a modelling failure."
                )
            elif diff > 0:
                st.success(f"Holt's Exponential Smoothing outperformed the naive moving-average baseline by {diff:.2f} incidents/month on this backtest.")
            else:
                st.warning(f"The naive moving-average baseline actually outperformed Holt's Exponential Smoothing by {-diff:.2f} incidents/month on this backtest — worth keeping in mind given how little real trend this dataset contains.")
            chart_card_close()
        else:
            st.caption(f"Backtest not available: {backtest.get('reason', 'insufficient data')}")

    # -------------------------------------------------------------
    with tab4:
        st.subheader("Anomaly Detection (Isolation Forest)")
        anomalies = insights_df[insights_df["Anomaly"] == True].sort_values("Anomaly_Score")
        st.write(f"**{len(anomalies)} incidents** flagged as statistically unusual "
                 f"(out of {len(insights_df)}), based on Financial Impact, Risk Score, Likelihood, and Impact Score.")

        fig = px.scatter(insights_df, x="Financial_Impact_GBP", y="Risk_Score",
                          color="Anomaly", hover_data=["Incident_ID", "Category"],
                          color_discrete_map={True: COLORS["critical"], False: COLORS["muted"]})
        fig.update_layout(height=400, plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font_color=COLORS["text"], xaxis_gridcolor=COLORS["border"], yaxis_gridcolor=COLORS["border"], legend=dict(font=dict(color=COLORS["text"])))
        st.plotly_chart(fig, width='stretch')

        st.subheader("Key finding: rare-but-catastrophic risks can score 'Low'")
        low_tier_flagged = insights_df[(insights_df["Anomaly"] == True) &
                                        (insights_df["Risk_Tier"] == "Low") &
                                        (insights_df["Financial_Impact_GBP"] > 1_000_000)]
        st.markdown(f"""
        <div style="background:{COLORS['surface']}; border:1px solid {COLORS['critical']};
                    border-left:5px solid {COLORS['critical']}; border-radius:10px; padding:18px 20px;
                    margin-bottom:16px;">
            <div style="display:flex; align-items:baseline; gap:14px; margin-bottom:10px;">
                <span style="font-family:'Sora',sans-serif; font-size:2.2rem; font-weight:700;
                             color:{COLORS['critical']};">{len(low_tier_flagged)}</span>
                <span style="color:{COLORS['text']}; font-size:1rem; font-weight:600;">
                    incidents scored "Low" risk tier despite Critical severity and over £1m financial impact
                </span>
            </div>
            <div style="color:{COLORS['text']}; font-size:0.9rem; line-height:1.5;">
                All are in <b>Damage to Physical Assets</b> &mdash; a historically rare category. Because
                Likelihood and Impact are <i>multiplied</i> together, a rare category's low Likelihood score
                drags the combined Risk Score down even when the Impact itself is catastrophic. This is a
                genuine limitation of multiplicative RCSA-style scoring: rare "black swan" risks can be
                systematically underweighted, regardless of how severe a specific incident actually is.
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_html_table(
            low_tier_flagged[["Incident_ID", "Category", "Severity", "Financial_Impact_GBP", "Risk_Tier"]]
            .reset_index(drop=True),
            badge_col="Risk_Tier"
        )

# =======================================================================
# SCREEN 6: HELP
# =======================================================================
elif page == "Help":
    eyebrow("USER GUIDE")
    st.title("Help & Documentation")
    st.write(
        "This page explains how to use the Risk Monitor dashboard and what each screen shows you. "
        "It is written as internal documentation — expand any section below for details."
    )

    with st.expander("\U0001F680 Quick Start", expanded=True):
        st.write(
            "This dashboard helps you classify, score, and track operational risk incidents using an "
            "AI-assisted workflow. To get started, open **Submit Incident** from the sidebar and describe "
            "the incident in your own words. The system will automatically suggest a Basel risk category "
            "and calculate a risk score based on the incident's likely impact and how often similar "
            "incidents have occurred historically. You can accept the AI's suggestion or override it if "
            "you believe a different category applies. Once submitted, the incident is placed in a "
            "**Pending Approval** queue and a notification is sent to a senior risk officer for review "
            "before it is finalised. You can track any incident's status, or approve/reject incidents "
            "awaiting review, from the **Incident Detail** screen."
        )

    with st.expander("\U0001F4CA Main Dashboard"):
        st.write(
            "The Main Dashboard gives you a high-level view of the current incident landscape. The KPI "
            "cards at the top summarise the total number of incidents logged, how many currently carry a "
            "High or Critical risk tier, the average risk score across all incidents, how many remain Open "
            "or Under Investigation, and how many are awaiting approval. Below this, the **Incidents by "
            "Category** chart shows which types of risk event occur most frequently, and the **Risk Tier "
            "Breakdown** chart shows the overall balance of Low, Medium, High and Critical incidents — "
            "hover over either chart to see the underlying figures. The **Risk Trend Over Time** chart "
            "tracks how the average risk score has moved month to month, which can help identify whether "
            "risk exposure is rising or falling. Finally, the **Top 5 Highest Risk Incidents** table lists "
            "the incidents that currently warrant the most attention, ranked by risk score with financial "
            "impact and recency used as tie-breakers where scores are equal."
        )

    with st.expander("\U0001F4CB Incident List"):
        st.write(
            "The Incident List provides a searchable, filterable record of every incident logged in the "
            "system. You can narrow the list by Category or Risk Tier using the dropdown filters, or "
            "search the incident descriptions directly using the search box. Because the full incident log "
            "can be large, only the first 50 matching results are displayed at a time — use the filters to "
            "narrow your search further if you need to find a specific incident. Selecting an incident from "
            "the dropdown at the bottom of the page will take you to its full Incident Detail record."
        )

    with st.expander("\U0001F4DD Submit Incident"):
        st.write(
            "Use this screen to log a new operational risk incident. Describe what happened in the "
            "free-text box in your own words, in the same way you would write an internal incident report "
            "— the more detail you provide, the more accurately the system can classify it. Provide your "
            "own estimate of the incident's severity and, where known, its likely financial impact. When "
            "you click **Classify & Submit**, the system analyses the description using a trained "
            "text-classification model and returns its best-guess Basel risk category, along with a "
            "confidence breakdown showing how strongly it considered other possible categories. Review "
            "this suggestion: if you agree, you can confirm it as-is; if you believe a different category "
            "is more appropriate, you can override it using the dropdown before saving. Once confirmed, "
            "the incident is saved with a status of Pending Approval, and an email notification is sent to "
            "the designated senior risk officer so that it can be reviewed before being finalised."
        )

    with st.expander("\U0001F50D Incident Detail"):
        st.write(
            "This screen shows the complete record for a single incident, including its original "
            "description, the category assigned to it, its severity and financial impact, and a full "
            "breakdown of how its risk score was calculated. Where the AI model's confidence scores were "
            "recorded at submission, they are displayed as a bar chart so you can see how certain the "
            "system was, and which other categories it considered plausible. If the incident's status is "
            "Pending Approval, this screen is also where an authorised reviewer approves or rejects it — "
            "approving moves it to Open, ready for handling in the normal way, while rejecting flags it as "
            "Rejected for correction or further review. For incidents that have already been approved, "
            "this screen allows the status to be updated as work progresses, for example to Under "
            "Investigation, Escalated, Resolved or Closed."
        )

    with st.expander("\U0001F4A1 Insights"):
        st.write(
            "The Insights screen goes beyond individual incidents to highlight broader patterns across the "
            "whole incident log, organised into four tabs. **Root-Cause Themes** groups incidents with "
            "similar language into sub-categories that are more specific than the eight official risk "
            "categories, and pairs each theme with a recommended preventive action. **Association "
            "Patterns** looks for combinations of category, severity and department that occur together "
            "more often than would be expected by chance, which can point to structural risk factors worth "
            "investigating further. **Trend Forecast** projects incident volume and average risk score "
            "over the coming months, and is accompanied by a statistical test indicating whether any "
            "apparent trend is likely to be genuine rather than random variation. **Anomaly Detection** "
            "flags incidents whose risk profile is statistically unusual compared with the rest of the "
            "dataset — including cases where a rare but severe incident type may be under-scored by the "
            "standard Likelihood × Impact formula, which is worth a closer look regardless of its "
            "calculated tier."
        )

    with st.expander("\U0001F4D0 Understanding Your Risk Score"):
        st.write(
            "Every incident's Risk Score is calculated using the same approach used across established "
            "operational risk frameworks: Risk Score equals Likelihood multiplied by Impact, each scored "
            "on a scale of 1 to 5. The Impact Score is derived from the incident's severity, reflecting the "
            "scale of financial loss typically associated with incidents of that severity. The Likelihood "
            "Score reflects how often that category of incident has historically occurred relative to "
            "other categories — a category that occurs frequently receives a higher Likelihood Score than "
            "one that is rare. Multiplying the two together produces a combined score between 1 and 25, "
            "which is then translated into a Risk Tier: Low, Medium, High or Critical."
        )
        st.write(
            "For example, an incident with Impact Score 5 (Critical severity) but Likelihood Score 1 (a "
            "historically rare category) produces a Risk Score of 5, placing it in the Low tier — even "
            "though its individual impact is severe. This is a known characteristic of multiplicative "
            "Likelihood × Impact scoring, and is worth bearing in mind: a Low overall tier does not always "
            "mean an incident is safe to overlook, particularly if flagged separately by the Anomaly "
            "Detection tab."
        )

    with st.expander("\U0001F916 Understanding the AI's Predictions"):
        st.write(
            "The category suggested for each incident is produced by a machine learning model trained on "
            "a representative set of past incident descriptions. In testing, the model correctly "
            "classifies incidents into their proper category around 84% of the time — a solid result, but "
            "not infallible, and it will occasionally suggest a category that, on reflection, does not "
            "fit. This is why every prediction is shown alongside its confidence breakdown rather than "
            "presented as a definitive answer, and why the workflow always requires a human to confirm or "
            "override the suggested category before it is saved. If the confidence in the top suggested "
            "category is low, or closely matched by a second option, it is worth reading the incident "
            "description again before accepting the AI's suggestion."
        )

    with st.expander("\u2705 The Approval Workflow"):
        st.write(
            "Every newly submitted incident is treated as unconfirmed until a second, authorised reviewer "
            "has checked it — a standard control in operational risk management often referred to as a "
            "maker-checker or four-eyes process, intended to catch errors or misjudgements before they are "
            "acted upon. When an incident is submitted, it is saved with a status of Pending Approval and "
            "an email notification is sent automatically to the designated senior risk officer. That "
            "reviewer can then open the incident's Detail screen, review the description, category, and "
            "risk score, and either approve it — moving it to Open status for normal handling — or reject "
            "it, flagging it for correction or further discussion with the person who submitted it."
        )

    with st.expander("\U0001F4D6 Glossary"):
        glossary_items = [
            ("Basel Event Type", "One of the official regulatory categories of operational risk (e.g. Internal Fraud, External Fraud, Business Disruption & System Failures), plus Cybersecurity Incident as an additional category used in this system."),
            ("Risk Tier", "The Low / Medium / High / Critical label derived from an incident's combined Risk Score."),
            ("Likelihood Score", "A 1-5 score reflecting how frequently an incident's category has historically occurred."),
            ("Impact Score", "A 1-5 score reflecting the scale of financial/operational consequence, derived from Severity."),
            ("Maker-Checker", "A control where the person submitting an incident (the 'maker') is different from the person who approves it (the 'checker'), reducing the risk of unchecked errors."),
            ("Model Confidence", "The probability the AI model assigns to each possible category — higher confidence means the model is more certain of its top suggestion."),
        ]
        for term, definition in glossary_items:
            st.markdown(f"**{term}** — {definition}")