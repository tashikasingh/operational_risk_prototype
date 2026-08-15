"""
Risk Scoring Layer (v2 -- Noisy Dataset)
===========================================

UPDATE FROM v1
----------------
The v1 script derived Likelihood from the ORIGINAL 200-row dataset
specifically because the 500-row expanded dataset had been artificially
rebalanced toward equal category counts for classifier training
purposes. The new noisy 500-row dataset (built per supervisor feedback)
was NOT artificially rebalanced -- category counts range naturally from
45 (Damage to Physical Assets) to 71 (Internal Fraud) as a byproduct of
the noise-injection process, not a deliberate equalisation. This means
Likelihood can now be derived directly from THIS dataset's own category
frequency, without the earlier original-vs-rebalanced distinction.

METHODOLOGY (unchanged from v1, for Methodology chapter)
------------------------------------------------------------
Risk Score = Likelihood x Impact (RCSA-style, ISO 31000 / COSO
practice; see v1 risk_scoring.py docstring for full literature
justification).

IMPACT SCORE (1-5): from Severity (Low=1, Medium=2, High=3, Critical=5).

LIKELIHOOD SCORE (1-5): linear scale based on this dataset's own
category frequency (min count -> 1, max count -> 5).

OUTPUT
------
data/processed/incidents_with_risk_scores.csv
"""

import pandas as pd
import json

df = pd.read_excel(
    "data/processed/operational_risk_synthetic_dataset_noisy.xlsx",
    sheet_name="Incident_Data"
)

category_counts = df["Category"].value_counts()
print("Category frequency (basis for Likelihood):")
print(category_counts.sort_values())

min_count = category_counts.min()
max_count = category_counts.max()

def likelihood_from_count(count):
    if max_count == min_count:
        return 3
    scaled = 1 + 4 * (count - min_count) / (max_count - min_count)
    return round(scaled)

category_likelihood = {cat: likelihood_from_count(cnt) for cat, cnt in category_counts.items()}

print("\nLikelihood score per category (1-5):")
for cat, score in sorted(category_likelihood.items(), key=lambda x: x[1]):
    print(f"  {cat}: {score}")

severity_to_impact = {"Low": 1, "Medium": 2, "High": 3, "Critical": 5}

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

print("\n--- Risk Tier Distribution ---")
print(df["Risk_Tier"].value_counts())

print("\n--- Sample scored incidents ---")
sample_cols = ["Incident_ID", "Category", "Severity", "Financial_Impact_GBP",
               "Likelihood_Score", "Impact_Score", "Risk_Score", "Risk_Tier"]
print(df[sample_cols].sample(8, random_state=42).to_string(index=False))

OUTPUT_PATH = "data/processed/incidents_with_risk_scores.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")

# ---------------------------------------------------------------------
# Save scoring lookup tables so the dashboard applies IDENTICAL logic
# to newly submitted incidents, rather than recomputing likelihood from
# a live (and potentially skewed, if dashboard-added incidents shift
# the distribution) category count each time.
# ---------------------------------------------------------------------
scoring_lookup = {
    "category_likelihood": category_likelihood,
    "severity_to_impact": severity_to_impact,
}
with open("models/risk_scoring_lookup.json", "w") as f:
    json.dump(scoring_lookup, f, indent=2)
print("Saved: models/risk_scoring_lookup.json")