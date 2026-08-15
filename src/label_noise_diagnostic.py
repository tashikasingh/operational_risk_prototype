"""
Label Noise Diagnostic
=========================

PURPOSE
-------
For the 26 incidents where the recorded Category deliberately differs
from True_Category (simulating realistic human labelling error), this
script checks whether the trained classifier's predictions tend to
align with the RECORDED (noisy) label, the TRUE label, or neither.

WHY THIS MATTERS (for Results/Discussion chapter)
------------------------------------------------------
If the model's predictions lean toward True_Category more often than
chance, this suggests the classifier is picking up genuine linguistic
signal in the incident description that reflects the actual underlying
event -- rather than simply memorising whatever label it was given
during training. This would be a positive, discussable finding:
evidence the model has learned meaningful category-distinguishing
language patterns, not just label associations.

If instead predictions lean toward the recorded (noisy) label, this
suggests the model is sensitive to label noise -- also a legitimate,
citable finding, consistent with literature on label noise in text
classification (Rączkowska et al., 2024; Jiang et al., 2024).

IMPORTANT METHODOLOGICAL NOTE
--------------------------------
This diagnostic runs the trained model on ALL 500 rows (including ones
used in training), which is NOT how we evaluate overall accuracy --
that remains validated only on the held-out test set (see
train_classifier.py). This script exists purely to inspect behaviour
on the specific 26 known label-noise cases, regardless of which side
of the train/test split they fell on. It should not be used, and is
not used elsewhere in this project, as a substitute for proper
held-out evaluation.

OUTPUT
------
Printed: summary counts + full table of the 26 label-noise cases with
their recorded label, true label, and model prediction.
data/processed/label_noise_diagnostic.csv
"""

import pandas as pd
import joblib

df = pd.read_excel(
    "data/processed/operational_risk_synthetic_dataset_noisy.xlsx",
    sheet_name="Incident_Data"
)
df["Description_clean"] = df["Description"].str.strip().str.lower()

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/naive_bayes_model.pkl")

X_all = vectorizer.transform(df["Description_clean"])
df["Model_Prediction"] = model.predict(X_all)

# ---------------------------------------------------------------------
# Isolate the label-noise cases
# ---------------------------------------------------------------------
noise_cases = df[df["Category"] != df["True_Category"]].copy()
print(f"Total label-noise cases: {len(noise_cases)}")

noise_cases["Matches_Recorded"] = noise_cases["Model_Prediction"] == noise_cases["Category"]
noise_cases["Matches_True"] = noise_cases["Model_Prediction"] == noise_cases["True_Category"]
noise_cases["Matches_Neither"] = ~noise_cases["Matches_Recorded"] & ~noise_cases["Matches_True"]

n_matches_recorded = noise_cases["Matches_Recorded"].sum()
n_matches_true = noise_cases["Matches_True"].sum()
n_matches_neither = noise_cases["Matches_Neither"].sum()

print(f"\nModel prediction matches RECORDED (noisy) label: {n_matches_recorded} ({n_matches_recorded/len(noise_cases)*100:.1f}%)")
print(f"Model prediction matches TRUE label:              {n_matches_true} ({n_matches_true/len(noise_cases)*100:.1f}%)")
print(f"Model prediction matches NEITHER:                 {n_matches_neither} ({n_matches_neither/len(noise_cases)*100:.1f}%)")

if n_matches_true > n_matches_recorded:
    print("\n--> Model leans toward TRUE category more often than the recorded label.")
    print("    This suggests the classifier is picking up genuine linguistic signal")
    print("    reflecting the actual event, rather than memorising noisy labels.")
elif n_matches_recorded > n_matches_true:
    print("\n--> Model leans toward the RECORDED (noisy) label more often.")
    print("    This suggests some sensitivity to label noise during training --")
    print("    a legitimate, citable limitation (cf. Rączkowska et al., 2024;")
    print("    Jiang et al., 2024, on label noise in text classification).")
else:
    print("\n--> No clear lean either way.")

# ---------------------------------------------------------------------
# Full table for inspection / appendix
# ---------------------------------------------------------------------
display_cols = ["Incident_ID", "Description", "Category", "True_Category",
                 "Model_Prediction", "Matches_Recorded", "Matches_True"]
print("\n--- Full label-noise case table ---")
print(noise_cases[display_cols].to_string(index=False))

noise_cases[display_cols].to_csv("data/processed/label_noise_diagnostic.csv", index=False)
print("\nSaved: data/processed/label_noise_diagnostic.csv")