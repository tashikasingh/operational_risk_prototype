"""
Insights Analysis
====================

PURPOSE
-------
Goes beyond classification + risk scoring to answer "why do these
incidents happen, and how could they be prevented" -- produces four
analyses consumed by the dashboard's Insights screen:

1. ROOT-CAUSE TOPIC CLUSTERING (K-Means on TF-IDF)
   Groups incidents into finer-grained sub-themes than the 8 Basel
   categories alone (e.g. within Internal Fraud: "rogue trading /
   limit breaches" vs "expense claim fraud" are genuinely different
   root causes needing different controls).

2. ASSOCIATION RULE MINING (Apriori)
   Looks for genuine co-occurrence patterns between Category, Severity,
   and Department. IMPORTANT: Detection_Method is deliberately EXCLUDED
   from this analysis -- it is near-deterministic per category in this
   synthetic dataset (a generation artifact, not a real-world pattern),
   confirmed via crosstab inspection. Including it would produce
   impressive-looking but methodologically meaningless "insights".

3. TREND FORECASTING (Holt's Exponential Smoothing)
   Forecasts incident volume and average risk score for the next 3
   months. A linear-regression significance test is run FIRST -- if
   p > 0.05 (as found here: incident count p=0.59, risk score p=0.32),
   this is reported explicitly as "no statistically significant trend
   detected", consistent with dates being randomly assigned during
   dataset generation. The forecast numbers are shown, but the
   dashboard/report must not overstate them as a meaningful prediction.

4. ANOMALY DETECTION (Isolation Forest)
   Flags incidents whose combination of Financial_Impact_GBP, Risk_Score,
   Likelihood_Score, and Impact_Score is statistically unusual --
   surfaces cases like a Critical-severity, high financial impact
   incident that still scores "Low" risk tier due to its category's
   rarity, a genuine and citable critique of multiplicative RCSA scoring.

5. PREVENTION RECOMMENDATIONS (rule-based, keyword-triggered)
   Maps each cluster's dominant theme to a suggested control, drawing
   on standard preventive/detective/corrective control categories
   (COSO / Basel AMA control practice). This is explicitly a rule-based
   lookup, not a black-box AI output -- kept transparent and defensible.

OUTPUTS (all in data/processed/)
------------------------------------
incidents_with_insights.csv   -- full data + Cluster, Anomaly columns
cluster_summary.json          -- cluster labels, themes, recommendations
association_rules.csv
trend_forecast.json
"""

import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats

df = pd.read_csv("data/processed/incidents_with_risk_scores.csv")
df["Description_clean"] = df["Description"].str.strip().str.lower()
print(f"Loaded {len(df)} incidents")

# =======================================================================
# 1. ROOT-CAUSE TOPIC CLUSTERING
# =======================================================================
print("\n=== 1. Root-Cause Topic Clustering ===")

vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2), min_df=3)
X = vectorizer.fit_transform(df["Description_clean"])

N_CLUSTERS = 12
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df["Cluster"] = km.fit_predict(X)

terms = vectorizer.get_feature_names_out()

def dedup_top_terms(center, terms, n=3):
    """Picks top terms by centroid weight, skipping terms that are
    substrings of an already-picked term (avoids redundant labels like
    'grievance filed' + 'grievance')."""
    order = center.argsort()[::-1]
    picked = []
    for i in order:
        term = terms[i]
        if not any(term in p or p in term for p in picked):
            picked.append(term)
        if len(picked) == n:
            break
    return picked

# Rule-based prevention recommendations, keyed by keyword triggers found
# in a cluster's top terms. Falls back to a generic category-level
# recommendation if no specific keyword matches.
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

cluster_summary = []
for c in range(N_CLUSTERS):
    mask = df["Cluster"] == c
    size = mask.sum()
    if size == 0:
        continue
    top_terms = dedup_top_terms(km.cluster_centers_[c], terms, n=3)
    dominant_cat = df[mask]["Category"].mode()[0]
    label = " / ".join(top_terms)

    recommendation = None
    for keywords, rec in KEYWORD_RECOMMENDATIONS:
        if any(any(kw in t for t in top_terms) for kw in keywords):
            recommendation = rec
            break
    if recommendation is None:
        recommendation = f"Review controls specific to {dominant_cat} incidents; no specific keyword-matched recommendation for this cluster's theme."

    cluster_summary.append({
        "cluster_id": int(c),
        "label": label,
        "size": int(size),
        "dominant_category": dominant_cat,
        "top_terms": top_terms,
        "recommendation": recommendation,
    })
    print(f"Cluster {c} (n={size}, {dominant_cat}): {label}")

with open("data/processed/cluster_summary.json", "w") as f:
    json.dump(cluster_summary, f, indent=2)

# =======================================================================
# 2. ASSOCIATION RULE MINING (Category, Severity, Department ONLY --
#    Detection_Method excluded, see docstring)
# =======================================================================
print("\n=== 2. Association Rule Mining ===")

transactions = []
for _, row in df.iterrows():
    items = [f"Category={row['Category']}", f"Severity={row['Severity']}"]
    if pd.notna(row["Department"]):
        items.append(f"Department={row['Department']}")
    transactions.append(items)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
trans_df = pd.DataFrame(te_ary, columns=te.columns_)

freq_items = apriori(trans_df, min_support=0.02, use_colnames=True)
rules = association_rules(freq_items, metric="lift", min_threshold=1.3)
rules = rules[rules["confidence"] >= 0.4].sort_values("lift", ascending=False)
rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(x))
rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(x))
rules_out = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
rules_out.to_csv("data/processed/association_rules.csv", index=False)
print(f"Found {len(rules_out)} genuine association rules (Detection_Method excluded as a dataset artifact)")
print(rules_out.to_string())

# =======================================================================
# 3. TREND FORECASTING
# =======================================================================
print("\n=== 3. Trend Forecasting ===")

df["Date_Reported"] = pd.to_datetime(df["Date_Reported"], errors="coerce")
trend_df = df.dropna(subset=["Date_Reported"]).copy()
trend_df["Month"] = trend_df["Date_Reported"].dt.to_period("M").astype(str)
monthly = trend_df.groupby("Month").agg(
    Incident_Count=("Incident_ID", "count"),
    Avg_Risk_Score=("Risk_Score", "mean")
).reset_index()

x = np.arange(len(monthly))
slope_count, _, r_count, p_count, _ = stats.linregress(x, monthly["Incident_Count"])
slope_risk, _, r_risk, p_risk, _ = stats.linregress(x, monthly["Avg_Risk_Score"])

print(f"Incident count trend: slope={slope_count:.4f}, p={p_count:.4f} "
      f"({'SIGNIFICANT' if p_count < 0.05 else 'NOT statistically significant'})")
print(f"Risk score trend: slope={slope_risk:.4f}, p={p_risk:.4f} "
      f"({'SIGNIFICANT' if p_risk < 0.05 else 'NOT statistically significant'})")

count_model = ExponentialSmoothing(monthly["Incident_Count"], trend="add", seasonal=None).fit()
count_forecast = count_model.forecast(3).tolist()

risk_model = ExponentialSmoothing(monthly["Avg_Risk_Score"], trend="add", seasonal=None).fit()
risk_forecast = risk_model.forecast(3).tolist()

trend_output = {
    "historical": monthly.to_dict(orient="records"),
    "forecast_incident_count": count_forecast,
    "forecast_avg_risk_score": risk_forecast,
    "incident_count_trend_significant": bool(p_count < 0.05),
    "incident_count_p_value": float(p_count),
    "risk_score_trend_significant": bool(p_risk < 0.05),
    "risk_score_p_value": float(p_risk),
}
with open("data/processed/trend_forecast.json", "w") as f:
    json.dump(trend_output, f, indent=2)

# =======================================================================
# 4. ANOMALY DETECTION
# =======================================================================
print("\n=== 4. Anomaly Detection ===")

features = df[["Financial_Impact_GBP", "Risk_Score", "Likelihood_Score", "Impact_Score"]]
iso = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = (iso.fit_predict(features) == -1)
df["Anomaly_Score"] = iso.decision_function(features)

n_anomalies = df["Anomaly"].sum()
print(f"Flagged {n_anomalies} anomalies out of {len(df)} incidents")

# Highlight the "rare-but-catastrophic scored Low" tension specifically
low_tier_high_impact = df[(df["Anomaly"]) & (df["Risk_Tier"] == "Low") & (df["Financial_Impact_GBP"] > 1_000_000)]
print(f"Anomalies that are 'Low' risk tier despite >£1m impact: {len(low_tier_high_impact)}")
if len(low_tier_high_impact) > 0:
    print(low_tier_high_impact[["Incident_ID", "Category", "Severity", "Financial_Impact_GBP", "Risk_Tier"]].to_string())

# =======================================================================
# Save consolidated incidents file (with Cluster + Anomaly columns)
# =======================================================================
df.to_csv("data/processed/incidents_with_insights.csv", index=False)
print("\nSaved: data/processed/incidents_with_insights.csv")
print("Saved: data/processed/cluster_summary.json")
print("Saved: data/processed/association_rules.csv")
print("Saved: data/processed/trend_forecast.json")