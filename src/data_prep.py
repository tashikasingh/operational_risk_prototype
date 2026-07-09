"""
Data Preparation Script
========================

PURPOSE
-------
Loads the expanded 500-row operational risk dataset, checks it for quality
issues, and produces a stratified train/test split ready for the text
classifier (TF-IDF + Naive Bayes / Logistic Regression / Random Forest).

WHY THESE STEPS (for dissertation Methodology chapter)
--------------------------------------------------------
1. STRATIFIED SPLIT: With 8 categories at ~62-63 rows each, a plain random
   80/20 split could by chance under- or over-represent a category in the
   test set. Stratified splitting (via scikit-learn's `stratify` parameter)
   splits EACH category 80/20 individually, so every category keeps its
   proportional representation in both train and test sets. This is
   standard practice for multi-class classification and gives more
   reliable per-class evaluation metrics later.

2. FIXED RANDOM SEED (random_state=42): ensures the split is reproducible —
   running this script twice gives the exact same train/test split, which
   matters for consistent, comparable results across your dissertation
   write-up and for anyone (e.g. an examiner) re-running your code.

3. TEXT CLEANING (lowercasing, whitespace normalisation): reduces
   unnecessary vocabulary size for the TF-IDF step later. E.g. without
   lowercasing, "Fraud" and "fraud" would be treated as two different
   words, artificially inflating the vocabulary and diluting word
   frequency signals.

4. SAVING THE SPLIT TO DISK: train.csv and test.csv are saved once here,
   so later scripts (train_classifier.py, evaluate.py) load the SAME split
   every time rather than re-splitting randomly on each run — this is
   what makes the whole pipeline reproducible end to end.

OUTPUT
------
data/processed/train.csv
data/processed/test.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
INPUT_PATH = "data/processed/operational_risk_synthetic_dataset_expanded.xlsx"
TRAIN_OUTPUT = "data/processed/train.csv"
TEST_OUTPUT = "data/processed/test.csv"

df = pd.read_excel(INPUT_PATH, sheet_name="Incident_Data")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------
# 2. Data quality checks
# ---------------------------------------------------------------------
print("\n--- Data Quality Checks ---")

# Missing values
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "  None found")

# Duplicate rows (exact duplicates across all columns)
n_dupes = df.duplicated().sum()
print(f"\nExact duplicate rows: {n_dupes}")

# Duplicate descriptions (same text, different incident — worth knowing,
# not necessarily a problem, since templated generation can produce
# near-identical phrasing for different entities)
n_dupe_desc = df["Description"].duplicated().sum()
print(f"Duplicate description text: {n_dupe_desc}")

# Empty or very short descriptions (would carry little signal for TF-IDF)
short_desc = df[df["Description"].str.len() < 15]
print(f"Descriptions under 15 characters: {len(short_desc)}")

# Category counts (confirm balance after expansion)
print("\nCategory distribution:")
print(df["Category"].value_counts())

# ---------------------------------------------------------------------
# 3. Basic text cleaning
# ---------------------------------------------------------------------
# Lowercase and strip extra whitespace. We deliberately do NOT remove
# punctuation or numbers at this stage -- TF-IDF handles tokenisation,
# and numbers (e.g. customer counts) may carry weak signal. Aggressive
# cleaning happens inside the TF-IDF vectoriser step later if needed.
df["Description_clean"] = (
    df["Description"]
    .str.strip()
    .str.lower()
)

# ---------------------------------------------------------------------
# 4. Select columns needed for classification
# ---------------------------------------------------------------------
# We keep the ID and original columns too (useful for error analysis
# later -- e.g. looking up which specific incidents were misclassified)
model_df = df[[
    "Incident_ID", "Description", "Description_clean",
    "Category", "Severity", "Financial_Impact_GBP"
]].copy()

# ---------------------------------------------------------------------
# 5. Stratified train/test split
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

# ---------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------
train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)
print(f"\nSaved: {TRAIN_OUTPUT}")
print(f"Saved: {TEST_OUTPUT}")