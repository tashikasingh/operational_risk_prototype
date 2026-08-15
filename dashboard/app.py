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
    "bg": "#F4F5F9", "surface": "#FFFFFF", "navy": "#131C34", "navy_light": "#1E2A4A",
    "accent": "#3B5BA9", "text": "#1C2331", "muted": "#6B7280",
    "low": "#2E9E5B", "medium": "#E3A83B", "high": "#E2793A", "critical": "#D64545",
}

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {COLORS['text']}; }}
    .stApp {{ background-color: {COLORS['bg']}; }}
    h1, h2, h3 {{ font-family: 'Sora', sans-serif !important; color: {COLORS['navy']} !important; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['navy']};
    }}
    section[data-testid="stSidebar"] * {{ color: #E8EAF0 !important; }}
    section[data-testid="stSidebar"] h1 {{
        font-family: 'Sora', sans-serif !important; font-size: 1.3rem !important;
        color: #FFFFFF !important; border-bottom: 1px solid #2A3A5C; padding-bottom: 12px;
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
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em;
        color: {COLORS['accent']}; text-transform: uppercase; margin-bottom: -6px; font-weight: 500;
    }}

    /* KPI cards */
    .kpi-row {{ display: flex; gap: 16px; margin: 8px 0 20px 0; flex-wrap: wrap; }}
    .kpi-card {{
        background: {COLORS['surface']}; border-radius: 12px; padding: 16px 20px; flex: 1; min-width: 180px;
        box-shadow: 0 1px 3px rgba(20,25,50,0.08); border-left: 4px solid var(--accent-color);
    }}
    .kpi-icon {{ font-size: 1.3rem; margin-bottom: 4px; }}
    .kpi-label {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: {COLORS['muted']};
        text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .kpi-value {{ font-family: 'Sora', sans-serif; font-size: 1.9rem; font-weight: 700; color: {COLORS['navy']}; }}

    /* Risk tier badges -- the signature element, used everywhere */
    .badge {{
        display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.78rem;
        font-weight: 600; font-family: 'Inter', sans-serif; color: white;
    }}
    .badge-low {{ background-color: {COLORS['low']}; }}
    .badge-medium {{ background-color: {COLORS['medium']}; }}
    .badge-high {{ background-color: {COLORS['high']}; }}
    .badge-critical {{ background-color: {COLORS['critical']}; }}

    /* Custom HTML tables */
    .risk-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; background: {COLORS['surface']};
        border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(20,25,50,0.08); }}
    .risk-table th {{
        background-color: {COLORS['navy']}; color: #E8EAF0 !important; text-align: left; padding: 10px 12px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .risk-table td {{ padding: 9px 12px; border-bottom: 1px solid #EEF0F5; }}
    .risk-table tr:last-child td {{ border-bottom: none; }}
    .risk-table tr:hover td {{ background-color: #F7F9FC; }}
    .incident-id {{ font-family: 'IBM Plex Mono', monospace; color: {COLORS['accent']}; font-weight: 500; }}

    /* Cards for cluster/theme lists etc */
    .theme-card {{
        background: {COLORS['surface']}; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
        border-left: 4px solid {COLORS['accent']}; box-shadow: 0 1px 3px rgba(20,25,50,0.06);
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['navy']}; color: white; border-radius: 8px; border: none;
        font-family: 'Inter', sans-serif; font-weight: 600;
    }}
    .stButton > button:hover {{ background-color: {COLORS['accent']}; color: white; }}
    </style>
    """, unsafe_allow_html=True)

BADGE_CLASS = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high", "Critical": "badge-critical"}

def risk_badge(tier):
    cls = BADGE_CLASS.get(tier, "badge-medium")
    return f'<span class="badge {cls}">{tier}</span>'

def kpi_card(icon, label, value, accent):
    return f"""<div class="kpi-card" style="--accent-color:{accent}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>"""

def render_kpis(cards):
    html = '<div class="kpi-row">' + "".join(cards) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_html_table(df, badge_col=None, id_col="Incident_ID"):
    """Renders a DataFrame as a styled HTML table with risk-tier badges,
    avoiding st.dataframe/st.table entirely (side-steps pyarrow dependency
    issues and gives full control over the visual design)."""
    df = df.copy()
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
    "\U0001F50D  Incident Detail", "\U0001F4A1  Insights"
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
    render_kpis([
        kpi_card("\U0001F4C1", "Total Incidents", len(df), COLORS["accent"]),
        kpi_card("\u26A0\ufe0f", "High / Critical Risk", len(df[df["Risk_Tier"].isin(["High", "Critical"])]), COLORS["high"]),
        kpi_card("\U0001F4C8", "Average Risk Score", f"{df['Risk_Score'].mean():.1f}", COLORS["medium"]),
        kpi_card("\U0001F550", "Open / Under Investigation", n_open, COLORS["navy"]),
    ])

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Incidents by Category")
        cat_counts = df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.bar(cat_counts, x="Category", y="Count")
        fig.update_traces(marker_color=COLORS["accent"])
        fig.update_layout(xaxis_tickangle=-30, height=400, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Risk Tier Breakdown")
        tier_counts = df["Risk_Tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk_Tier", "Count"]
        fig = px.pie(tier_counts, names="Risk_Tier", values="Count",
                     color="Risk_Tier",
                     color_discrete_map={"Low": COLORS["low"], "Medium": COLORS["medium"],
                                         "High": COLORS["high"], "Critical": COLORS["critical"]})
        fig.update_layout(height=400, paper_bgcolor="white")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Risk Trend Over Time")
    df["Date_Reported"] = pd.to_datetime(df["Date_Reported"], errors="coerce")
    trend = df.dropna(subset=["Date_Reported"]).copy()
    trend["Month"] = trend["Date_Reported"].dt.to_period("M").astype(str)
    monthly = trend.groupby("Month")["Risk_Score"].mean().reset_index()
    fig = px.line(monthly, x="Month", y="Risk_Score", markers=True)
    fig.update_traces(line_color=COLORS["accent"], marker_color=COLORS["navy"])
    fig.update_layout(height=350, yaxis_title="Average Risk Score", plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Top 5 Highest Risk Incidents")
    top5 = df.sort_values("Risk_Score", ascending=False).head(5)[
        ["Incident_ID", "Description", "Category", "Risk_Score", "Risk_Tier"]].copy()
    top5["Description"] = top5["Description"].str[:70] + "..."
    render_html_table(top5, badge_col="Risk_Tier")

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
    display_df["Description"] = display_df["Description"].str[:60] + "..."

    render_html_table(display_df.head(50).reset_index(drop=True), badge_col="Risk_Tier")

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
        fig.update_layout(height=250, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            final_category = st.selectbox("Confirm or override category",
                                            [p["predicted_category"]] +
                                            [c for c in category_likelihood if c != p["predicted_category"]])
        with c2:
            if st.button("Confirm & Save"):
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
                )
                st.success(f"Saved as {new_id}")
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
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, width='stretch')

        with c2:
            st.write(f"**Likelihood Score:** {record['Likelihood_Score']} / 5")
            st.write(f"**Impact Score:** {record['Impact_Score']} / 5")
            st.write(f"**Risk Score:** {record['Likelihood_Score']} × {record['Impact_Score']} = {record['Risk_Score']}")
            st.write(f"**Risk Tier:**")
            st.markdown(risk_badge(record['Risk_Tier']), unsafe_allow_html=True)

            new_status = st.selectbox(
                "Status", ["Open", "Under Investigation", "Escalated", "Resolved", "Closed"],
                index=["Open", "Under Investigation", "Escalated", "Resolved", "Closed"].index(record["Status"])
                if record["Status"] in ["Open", "Under Investigation", "Escalated", "Resolved", "Closed"] else 0
            )
            if st.button("Update Status"):
                db.update_status(incident_id, new_status)
                st.success("Status updated.")

# =======================================================================
# SCREEN 5: INSIGHTS (root-cause analysis, trends, anomalies)
# =======================================================================
elif page == "Insights":
    eyebrow("ROOT-CAUSE ANALYSIS")
    st.title("Insights")
    st.caption("Why incidents happen, and what to do about it.")
    st.caption("Goes beyond classification to surface root causes, patterns, trends, and anomalies.")

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
        fig = px.bar(cluster_df, x="Count", y="Theme", orientation="h", color="Dominant Category")
        fig.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')

        st.subheader("Recommended Actions per Root-Cause Theme")
        for c in clusters_sorted:
            with st.container(border=True):
                st.write(f"**{c['label']}** ({c['size']} incidents, mostly {c['dominant_category']})")
                st.write(f"→ {c['recommendation']}")

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
        fig.update_layout(height=350)
        st.plotly_chart(fig, width='stretch')

        sig_count = trend["incident_count_trend_significant"]
        st.write(f"**Trend significance:** p = {trend['incident_count_p_value']:.3f} — "
                 f"{'a statistically significant trend was found' if sig_count else 'no statistically significant trend detected (p > 0.05)'}")
        if not sig_count:
            st.caption("This is expected: incident dates were randomly assigned during synthetic data "
                       "generation, so no genuine temporal trend exists. A real bank's data would likely "
                       "show genuine seasonal/business-driven trends this synthetic dataset doesn't model.")

        st.write("**Next 3 months forecast (incident count):**", [round(v, 1) for v in trend["forecast_incident_count"]])
        st.write("**Next 3 months forecast (avg risk score):**", [round(v, 1) for v in trend["forecast_avg_risk_score"]])

    # -------------------------------------------------------------
    with tab4:
        st.subheader("Anomaly Detection (Isolation Forest)")
        anomalies = insights_df[insights_df["Anomaly"] == True].sort_values("Anomaly_Score")
        st.write(f"**{len(anomalies)} incidents** flagged as statistically unusual "
                 f"(out of {len(insights_df)}), based on Financial Impact, Risk Score, Likelihood, and Impact Score.")

        fig = px.scatter(insights_df, x="Financial_Impact_GBP", y="Risk_Score",
                          color="Anomaly", hover_data=["Incident_ID", "Category"],
                          color_discrete_map={True: "red", False: "lightgray"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

        st.subheader("Key finding: rare-but-catastrophic risks can score 'Low'")
        low_tier_flagged = insights_df[(insights_df["Anomaly"] == True) &
                                        (insights_df["Risk_Tier"] == "Low") &
                                        (insights_df["Financial_Impact_GBP"] > 1_000_000)]
        st.warning(
            f"{len(low_tier_flagged)} anomalous incidents have **Critical severity and >£1m financial "
            f"impact, yet score 'Low' risk tier** — all in Damage to Physical Assets, a historically rare "
            f"category. This is a genuine limitation of multiplicative Likelihood × Impact scoring: rare "
            f"categories get a low Likelihood weighting even when a specific incident is catastrophic, "
            f"potentially underweighting 'black swan' style risks."
        )
        render_html_table(
            low_tier_flagged[["Incident_ID", "Category", "Severity", "Financial_Impact_GBP", "Risk_Tier"]]
            .reset_index(drop=True),
            badge_col="Risk_Tier"
        )