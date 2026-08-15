"""
Data Preparation Script (v2 -- Noisy Dataset)
================================================

UPDATE FROM v1
----------------
Following supervisor feedback, the classifier trained on an overly
clean, templated synthetic dataset achieved a suspicious 100% test
accuracy. This version switches to a deliberately noisy 500-row
dataset incorporating: character-level typos, casing/punctuation
irregularities, ambiguous cross-category descriptions, deliberate
label noise (Category vs True_Category), near-duplicate descriptions
reused across categories, and vague/truncated text.

IMPORTANT METHODOLOGICAL DECISION: TRAIN ON 'Category', NOT 'True_Category'
-------------------------------------------------------------------------------
The classifier is trained and evaluated against the 'Category' field
(the recorded/noisy label), NOT 'True_Category'. This is deliberate:
in a real bank, the model would only ever see the label an analyst
actually recorded (which may itself contain human error) -- it would
never have access to a "ground truth" label. 'True_Category' is kept
in the output files purely as a DIAGNOSTIC field, to allow later
analysis of whether the model's misclassifications tend to align with
the true category (i.e. whether the model "sees through" label noise)
-- a genuinely interesting discussion point for the Results chapter,
not something to train on directly.

WHY WE DO NOT "CLEAN" THE TEXT NOISE
------------------------------------------
Unlike v1, we deliberately do NOT correct typos, casing, or truncation
in Description_clean beyond lowercasing. Removing the noise would
defeat the purpose of this dataset revision -- the classifier needs to
face realistic, imperfect text, since that is precisely what a
production incident-reporting system would produce.

OUTPUT
------
data/processed/train.csv  (includes Category, True_Category, noise flags)
data/processed/test.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/processed/operational_risk_synthetic_dataset_noisy.xlsx"
TRAIN_OUTPUT = "data/processed/train.csv"
TEST_OUTPUT = "data/processed/test.csv"

df = pd.read_excel(INPUT_PATH, sheet_name="Incident_Data")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------
# Data quality checks (expect noise indicators to show up here now --
# that's correct and expected, not a bug)
# ---------------------------------------------------------------------
print("\n--- Data Quality Checks ---")

missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "  None found")

n_dupes = df.duplicated().sum()
print(f"\nExact duplicate rows: {n_dupes}")

n_label_noise = (df["Category"] != df["True_Category"]).sum()
print(f"Label noise cases (Category != True_Category): {n_label_noise}")

n_ambiguous = (df["Is_Ambiguous_Case"] == "Yes").sum()
print(f"Ambiguous cross-category cases: {n_ambiguous}")

n_dup_text = (df["Is_Duplicate_Text"] == "Yes").sum()
print(f"Near-duplicate text cases: {n_dup_text}")

print("\nCategory distribution (recorded, i.e. what we train on):")
print(df["Category"].value_counts())

# ---------------------------------------------------------------------
# Minimal text cleaning -- lowercase + whitespace strip ONLY.
# Typos, casing noise, truncation are left intact deliberately (see docstring).
# ---------------------------------------------------------------------
df["Description_clean"] = (
    df["Description"]
    .str.strip()
    .str.lower()
)

# ---------------------------------------------------------------------
# Select columns -- keep True_Category and noise flags for diagnostics
# ---------------------------------------------------------------------
model_df = df[[
    "Incident_ID", "Description", "Description_clean",
    "Category", "True_Category", "Severity", "Financial_Impact_GBP",
    "Is_Ambiguous_Case", "Is_Label_Noise", "Is_Duplicate_Text",
]].copy()

# ---------------------------------------------------------------------
# Stratified train/test split on the RECORDED Category (what the
# classifier will actually be trained/evaluated on)
# ---------------------------------------------------------------------
train_df, test_df = train_test_split(
    model_df,
    test_size=0.2,
    stratify=model_df["Category"],
    random_state=42,
)

print(f"\nTrain set: {len(train_df)} rows")
print(train_df["Category"].value_counts())
print(f"\nTest set: {len(test_df)} rows")
print(test_df["Category"].value_counts())

train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)
print(f"\nSaved: {TRAIN_OUTPUT}")
print(f"Saved: {TEST_OUTPUT}")