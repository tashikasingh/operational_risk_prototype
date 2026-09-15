"""
Insights Engine (Shared Logic)
===================================

PURPOSE
-------
Extracts the four core analyses (root-cause clustering, association rule
mining, trend forecasting, anomaly detection) into reusable functions that
can run against ANY incidents DataFrame -- not just a static file.

WHY THIS MODULE EXISTS
---------------------------
The original insights_analysis.py was a one-shot script reading a fixed
Excel file. This meant incidents submitted through the dashboard (and
subsequently approved) never appeared in the Insights tab until someone
manually re-ran that script -- a real limitation, since the whole point
of a live dashboard is that it reflects current data. This module lets
the Insights page recompute against the LIVE database on demand (via a
"Refresh Insights" button), while insights_analysis.py can still import
these same functions for the original offline/reproducible analysis.

KNOWN LIMITATION -- DEPARTMENT NOT IN LIVE DATA
-----------------------------------------------------
The SQLite database schema (dashboard/db.py) does not currently store a
Department field -- it was never added to the Submit Incident form.
Association rule mining on LIVE data therefore uses Category + Severity
only, dropping Department from that specific analysis. The original
static analysis (run via insights_analysis.py on the full noisy dataset,
which DOES include Department) is unaffected. This is a deliberate,
disclosed simplification, not a silent gap.

FUNCTIONS
---------
run_clustering(df)           -> (df_with_cluster, cluster_summary_list)
run_association_rules(df)    -> rules_df (Category + Severity only)
run_trend_forecast(df)       -> trend_output dict
run_anomaly_detection(df)    -> df_with_anomaly
run_all_insights(df)         -> dict bundling all of the above
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats

KEYWORD_RECOMMENDATIONS = [
    (["ransomware", "malware", "exposed", "data"], "Strengthen data loss prevention (DLP) controls, endpoint protection, and breach notification protocols; conduct regular penetration testing."),
    (["unauthorised access", "unauthorised", "attack"], "Implement multi-factor authentication and continuous access monitoring for customer-facing systems."),
    (["grievance", "discriminatory", "complaint"], "Review HR grievance handling procedures and conduct workplace conduct/diversity training."),
    (["payment", "failure", "external"], "Review third-party vendor SLAs and implement payment processing redundancy/failover testing."),
    (["limits", "mismarked", "positions", "exceeded"], "Strengthen trade surveillance systems and enforce stricter segregation of duties on position limits."),
    (["suppression", "equipment", "branch"], "Increase facilities maintenance audit frequency and review insurance coverage for physical assets."),
    (["suitability", "inadequate", "regulatory"], "Enhance KYC/suitability assessment training and pre-sale product review checkpoints."),
    (["expense", "claims", "employee"], "Strengthen expense claim audit controls and automated anomaly flagging on reimbursement systems."),
    (["card", "fraud", "transactions"], "Enhance card-present fraud monitoring and enforce chip-and-PIN / tokenisation controls."),
    (["inspection", "staff", "office"], "Increase health & safety audit frequency across office/branch premises."),
    (["disrupted", "core banking", "outage", "platform"], "Invest in system resilience, redundancy, and regular failover/disaster-recovery testing for core banking systems."),
]


def _dedup_top_terms(center, terms, n=3):
    order = center.argsort()[::-1]
    picked = []
    for i in order:
        term = terms[i]
        if not any(term in p or p in term for p in picked):
            picked.append(term)
        if len(picked) == n:
            break
    return picked


def run_clustering(df, n_clusters=12, min_df=3):
    """Root-cause topic clustering via K-Means on TF-IDF description features."""
    n_clusters = min(n_clusters, max(2, len(df) // 8))  # guard against too few rows
    df = df.copy()
    df["Description_clean"] = df["Description"].fillna("").str.strip().str.lower()

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2),
                                  min_df=min(min_df, max(1, len(df) // 20)))
    X = vectorizer.fit_transform(df["Description_clean"])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(X)
    terms = vectorizer.get_feature_names_out()

    cluster_summary = []
    for c in range(n_clusters):
        mask = df["Cluster"] == c
        size = int(mask.sum())
        if size == 0:
            continue
        top_terms = _dedup_top_terms(km.cluster_centers_[c], terms, n=3)
        dominant_cat = df[mask]["Category"].mode()[0]
        label = " / ".join(top_terms)

        recommendation = None
        for keywords, rec in KEYWORD_RECOMMENDATIONS:
            if any(any(kw in t for t in top_terms) for kw in keywords):
                recommendation = rec
                break
        if recommendation is None:
            recommendation = f"Review controls specific to {dominant_cat} incidents."

        cluster_summary.append({
            "cluster_id": int(c), "label": label, "size": size,
            "dominant_category": dominant_cat, "top_terms": top_terms,
            "recommendation": recommendation,
        })
    return df, cluster_summary


def run_association_rules(df, min_support=0.02, min_lift=1.3, min_confidence=0.4):
    """Association rule mining on Category + Severity only (see module
    docstring -- Department is not available in live database records)."""
    transactions = []
    for _, row in df.iterrows():
        items = [f"Category={row['Category']}", f"Severity={row['Severity']}"]
        transactions.append(items)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_ary, columns=te.columns_)

    freq_items = apriori(trans_df, min_support=min_support, use_colnames=True)
    if len(freq_items) == 0:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    rules = rules[rules["confidence"] >= min_confidence].sort_values("lift", ascending=False)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(x))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(x))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)


def _run_backtest(monthly_for_stats, holdout=3):
    """Validates the forecasting approach against real data, rather than
    only trusting the forecast blindly. Holds out the last N complete
    months, fits BOTH Holt's Exponential Smoothing and a simple 3-month
    moving average baseline on the data before that, and compares each
    method's prediction against what actually happened in the held-out
    months. This directly answers 'how good is this forecast really?'
    rather than just showing a number with no way to judge it.

    A simple moving average is included as the baseline because if a
    sophisticated method like Holt's cannot beat a naive average of the
    last few months, that is itself an important, honest finding --
    consistent with this dataset's confirmed lack of a genuine trend
    (see incident_count_trend_significant above).
    """
    if len(monthly_for_stats) < holdout + 4:
        return {"available": False, "reason": "Not enough historical data for a reliable backtest."}

    train = monthly_for_stats.iloc[:-holdout].reset_index(drop=True)
    test = monthly_for_stats.iloc[-holdout:].reset_index(drop=True)

    # Holt's Exponential Smoothing, fit only on data before the held-out period
    holt_model = ExponentialSmoothing(train["Incident_Count"], trend="add", seasonal=None).fit()
    holt_pred = holt_model.forecast(holdout).tolist()

    # Simple moving average baseline: average of the last 3 training months,
    # repeated flat for each held-out month (the naive "no change expected" forecast)
    ma_value = float(train["Incident_Count"].tail(3).mean())
    ma_pred = [ma_value] * holdout

    actual = test["Incident_Count"].tolist()
    months = test["Month"].tolist()

    def mae(preds, actuals):
        return float(np.mean([abs(p - a) for p, a in zip(preds, actuals)]))

    return {
        "available": True,
        "months": months,
        "actual": actual,
        "holt_prediction": holt_pred,
        "moving_avg_prediction": ma_pred,
        "holt_mae": mae(holt_pred, actual),
        "moving_avg_mae": mae(ma_pred, actual),
    }


def run_trend_forecast(df):
    """Monthly trend aggregation + significance test + 3-month forecast.

    IMPORTANT: the CURRENT calendar month is excluded from the
    significance test and forecast (though still shown in the returned
    'historical' series for the chart). A month still in progress will
    always have an artificially low incident count simply because it
    hasn't finished yet -- not because incident volume is genuinely
    declining. Including it would create a fake "cliff" at the most
    recent data point, which can spuriously produce a "statistically
    significant" downward trend that has nothing to do with real risk
    patterns. This was caught after live incidents submitted through
    the dashboard (all dated with today's actual date) made the current,
    still-accumulating month look like a sharp drop-off.
    """
    df = df.copy()
    df["Date_Reported"] = pd.to_datetime(df["Date_Reported"], errors="coerce")
    trend_df = df.dropna(subset=["Date_Reported"])
    trend_df = trend_df.copy()
    trend_df["Month"] = trend_df["Date_Reported"].dt.to_period("M").astype(str)
    monthly = trend_df.groupby("Month").agg(
        Incident_Count=("Incident_ID", "count"),
        Avg_Risk_Score=("Risk_Score", "mean")
    ).reset_index()

    current_month = str(pd.Timestamp.now().to_period("M"))
    excluded_current_month = False
    monthly_for_stats = monthly
    if len(monthly) > 0 and monthly["Month"].iloc[-1] == current_month:
        monthly_for_stats = monthly.iloc[:-1].copy()
        excluded_current_month = True

    if len(monthly_for_stats) < 3:
        return {"historical": monthly.to_dict(orient="records"),
                "forecast_incident_count": [], "forecast_avg_risk_score": [],
                "incident_count_trend_significant": False, "incident_count_p_value": None,
                "risk_score_trend_significant": False, "risk_score_p_value": None,
                "excluded_current_month": excluded_current_month,
                "note": "Not enough COMPLETE monthly data points for a reliable forecast."}

    x = np.arange(len(monthly_for_stats))
    _, _, _, p_count, _ = stats.linregress(x, monthly_for_stats["Incident_Count"])
    _, _, _, p_risk, _ = stats.linregress(x, monthly_for_stats["Avg_Risk_Score"])

    count_model = ExponentialSmoothing(monthly_for_stats["Incident_Count"], trend="add", seasonal=None).fit()
    risk_model = ExponentialSmoothing(monthly_for_stats["Avg_Risk_Score"], trend="add", seasonal=None).fit()

    backtest = _run_backtest(monthly_for_stats)

    return {
        "historical": monthly.to_dict(orient="records"),
        "forecast_incident_count": count_model.forecast(3).tolist(),
        "forecast_avg_risk_score": risk_model.forecast(3).tolist(),
        "incident_count_trend_significant": bool(p_count < 0.05),
        "incident_count_p_value": float(p_count),
        "risk_score_trend_significant": bool(p_risk < 0.05),
        "risk_score_p_value": float(p_risk),
        "backtest": backtest,
        "excluded_current_month": excluded_current_month,
    }


def run_anomaly_detection(df, contamination=0.05):
    """Isolation Forest anomaly detection on risk/impact features."""
    df = df.copy()
    features = df[["Financial_Impact_GBP", "Risk_Score", "Likelihood_Score", "Impact_Score"]].fillna(0)
    iso = IsolationForest(contamination=contamination, random_state=42)
    df["Anomaly"] = (iso.fit_predict(features) == -1)
    df["Anomaly_Score"] = iso.decision_function(features)
    return df


def run_all_insights(df):
    """Runs all four analyses and returns a single bundled dict, ready
    for the dashboard to render or for saving to the cache files."""
    df_clustered, cluster_summary = run_clustering(df)
    rules_df = run_association_rules(df)
    trend = run_trend_forecast(df)
    df_final = run_anomaly_detection(df_clustered)

    return {
        "incidents_with_insights": df_final,
        "cluster_summary": cluster_summary,
        "association_rules": rules_df,
        "trend_forecast": trend,
    }