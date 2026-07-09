"""
Risk Scoring Layer
====================

PURPOSE
-------
Converts each incident's Category, Severity, and Financial_Impact_GBP into
a single numeric Risk Score and a Risk Tier label (Low/Medium/High/
Critical), following the industry-standard Risk & Control Self-Assessment
(RCSA) methodology used across ISO 31000, COSO, and banking operational
risk practice.

METHODOLOGY (for dissertation Methodology chapter)
------------------------------------------------------
Risk Score = Likelihood x Impact

This is the standard RCSA / risk-matrix formula, typically applied on a
5-point scale for each dimension, producing a combined score from 1-25 --
NOT a bespoke or arbitrary formula invented for this project. See:
  - RCSA practice: risk score calculated as Likelihood x Impact,
    aligning with organisational risk appetite (Wolters Kluwer, 2025)
  - 5x5 likelihood x impact matrices are the standard tool for producing
    a risk heat map (Onspring, 2025)

IMPACT SCORE (1-5)
--------------------
Derived from Severity, which we have already shown correlates with
distinct, non-overlapping Financial_Impact_GBP bands (see data
exploration in dataset expansion script). Mapping:
    Low = 1, Medium = 2, High = 3, Critical = 5
(Critical jumps to 5 rather than 4, reflecting the disproportionately
large financial gap between High (~£140k-495k) and Critical (~£1.4m-4.9m)
bands observed in the data.)

LIKELIHOOD SCORE (1-5)
-------------------------
IMPORTANT DESIGN DECISION: Likelihood is derived from category frequency
in the ORIGINAL 200-row dataset (OPR-0001-0200), NOT the expanded 500-row
dataset. This is deliberate: the 500-row dataset was rebalanced toward
~62-63 rows/category specifically to give the ML classifier enough
training examples per class -- that rebalancing was a modelling
convenience, not a claim about real-world incident base rates. Using the
rebalanced frequencies for Likelihood would incorrectly suggest all 8
categories are equally likely, undermining the purpose of a likelihood
score. The original 200 rows' natural (imbalanced) frequency distribution
is the more defensible proxy for genuine historical likelihood, in the
absence of real bank data or practitioner interviews (which would require
an E2 ethics amendment).

RISK TIER LABELS
------------------
Combined score (1-25) mapped to standard RCSA heat-map tiers:
    1-5   -> Low
    6-10  -> Medium
    11-15 -> High
    16-25 -> Critical
(Banding follows common 5x5 matrix conventions, e.g. Asana/Monday.com
risk matrix guides, adapted to this project's 1-25 scale.)

OUTPUT
------
data/processed/incidents_with_risk_scores.csv
"""

import pandas as pd

# ---------------------------------------------------------------------
# 1. Load expanded dataset (has all 500 rows) and identify original 200
#    rows specifically, to compute Likelihood from their natural frequency
# ---------------------------------------------------------------------
df = pd.read_excel(
    "data/processed/operational_risk_synthetic_dataset_expanded.xlsx",
    sheet_name="Incident_Data"
)

original_rows = df[df["Incident_ID"] <= "OPR-0200"]
category_counts = original_rows["Category"].value_counts()

print("Category frequency in ORIGINAL 200 rows (basis for Likelihood):")
print(category_counts.sort_values())

# ---------------------------------------------------------------------
# 2. Likelihood score (1-5): linear scale based on original category
#    frequency, min count -> 1, max count -> 5
# ---------------------------------------------------------------------
min_count = category_counts.min()
max_count = category_counts.max()

def likelihood_from_count(count):
    if max_count == min_count:
        return 3  # fallback if no variation
    scaled = 1 + 4 * (count - min_count) / (max_count - min_count)
    return round(scaled)

category_likelihood = {cat: likelihood_from_count(cnt) for cat, cnt in category_counts.items()}

print("\nLikelihood score per category (1-5):")
for cat, score in sorted(category_likelihood.items(), key=lambda x: x[1]):
    print(f"  {cat}: {score}")

# ---------------------------------------------------------------------
# 3. Impact score (1-5): from Severity
# ---------------------------------------------------------------------
severity_to_impact = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 5,
}

# ---------------------------------------------------------------------
# 4. Apply to full 500-row dataset
# ---------------------------------------------------------------------
df["Likelihood_Score"] = df["Category"].map(category_likelihood)
df["Impact_Score"] = df["Severity"].map(severity_to_impact)
df["Risk_Score"] = df["Likelihood_Score"] * df["Impact_Score"]

def risk_tier(score):
    if score <= 5:
        return "Low"
    elif score <= 10:
        return "Medium"
    elif score <= 15:
        return "High"
    else:
        return "Critical"

df["Risk_Tier"] = df["Risk_Score"].apply(risk_tier)

# ---------------------------------------------------------------------
# 5. Summary output
# ---------------------------------------------------------------------
print("\n--- Risk Tier Distribution (full 500-row dataset) ---")
print(df["Risk_Tier"].value_counts())

print("\n--- Sample scored incidents ---")
sample_cols = ["Incident_ID", "Category", "Severity", "Financial_Impact_GBP",
               "Likelihood_Score", "Impact_Score", "Risk_Score", "Risk_Tier"]
print(df[sample_cols].sample(8, random_state=42).to_string(index=False))

# ---------------------------------------------------------------------
# 6. Save enriched dataset
# ---------------------------------------------------------------------
OUTPUT_PATH = "data/processed/incidents_with_risk_scores.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")