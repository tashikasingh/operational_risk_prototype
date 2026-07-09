"""
Train Classifier — TF-IDF + Naive Bayes (baseline model)
==========================================================

PURPOSE
-------
Trains a text classifier that predicts the Basel risk Category directly
from the free-text incident Description. This is the first, simplest
model in the pipeline -- later scripts will add Logistic Regression and
Random Forest for comparison.

WHY TF-IDF + NAIVE BAYES AS THE FIRST MODEL (for Methodology chapter)
------------------------------------------------------------------------
- TF-IDF (Term Frequency-Inverse Document Frequency) converts each text
  description into a numeric vector, weighting words that are distinctive
  to a document (e.g. "ransomware", "phishing") higher than words that
  appear everywhere (e.g. "the", "customer").
- Multinomial Naive Bayes is a probabilistic classifier that works
  naturally with word-frequency-style features like TF-IDF. It assumes
  word occurrences are conditionally independent given the category --
  a simplifying assumption ("naive"), but one that performs surprisingly
  well for text classification in practice and is fast to train.
- Both are standard, well-established, interpretable baseline choices in
  NLP text classification literature, making them easy to justify and
  explain in a dissertation at this level -- as opposed to a deep learning
  model, which would be harder to justify given the dataset size (500
  rows) and harder to interpret/explain in a viva.

WHY WE FIT TF-IDF ON TRAINING DATA ONLY
-----------------------------------------
The vectoriser is `fit_transform`-ed on the TRAINING descriptions only,
then only `transform`-ed (not re-fit) on the test descriptions. This
prevents "data leakage" -- if the vectoriser learned vocabulary/weights
from the test set too, the model would have indirect knowledge of test
data it's supposed to be evaluated on, making the evaluation invalid.

WHY WE REPORT PER-CATEGORY METRICS, NOT JUST ACCURACY
---------------------------------------------------------
With 8 categories and some class imbalance in the ORIGINAL 200 rows
(now rebalanced to ~62-63/category), a model could achieve deceptively
high overall accuracy while performing poorly on specific categories.
`classification_report` gives precision, recall, and F1-score PER
CATEGORY, which is the honest way to evaluate a multi-class classifier.

WHY WE ALSO RUN 5-FOLD CROSS-VALIDATION
--------------------------------------------
A single 80/20 train/test split could, by chance, be an unusually easy
or unusually hard split. With only 500 rows, this risk is non-trivial.
5-fold stratified cross-validation retrains and re-evaluates the model
5 separate times, each holding out a different 20% slice as the test
fold, then reports the mean and standard deviation of accuracy across
all 5 runs. A low standard deviation (as found here) demonstrates the
reported accuracy is a stable property of the model/data, not a lucky
single split -- important evidence for the dissertation's Results
chapter when accuracy is very high, as it rules out "you just got a
lucky test set" as an explanation.

OUTPUTS
-------
models/tfidf_vectorizer.pkl   -- the fitted TF-IDF vectoriser
models/naive_bayes_model.pkl  -- the trained Naive Bayes classifier
Printed: accuracy, classification report, confusion matrix, CV scores
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------
# 1. Load train/test data
# ---------------------------------------------------------------------
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

X_train_text = train_df["Description_clean"]
y_train = train_df["Category"]

X_test_text = test_df["Description_clean"]
y_test = test_df["Category"]

# ---------------------------------------------------------------------
# 2. TF-IDF vectorisation
# ---------------------------------------------------------------------
# max_features caps vocabulary size to the N most informative words
# (by TF-IDF score) -- helps avoid overfitting on a small (500-row) dataset
# stop_words='english' removes common English words (the, is, at, ...)
# that carry no category-distinguishing signal
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words="english",
    ngram_range=(1, 2),  # unigrams AND bigrams, e.g. "data breach" as one feature
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)  # transform only, NOT fit -- see docstring

print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

# ---------------------------------------------------------------------
# 3. Train Naive Bayes
# ---------------------------------------------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 4. Predict and evaluate
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.3f}")

print("\n--- Per-Category Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n--- Confusion Matrix ---")
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# ---------------------------------------------------------------------
# 4b. 5-fold stratified cross-validation (robustness check)
# ---------------------------------------------------------------------
# Combine train+test back together for cross-validation, since CV does
# its own internal splitting across 5 folds. This gives a more robust
# accuracy estimate than a single 80/20 split alone (see docstring).
full_df = pd.concat([train_df, test_df], ignore_index=True)

cv_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))),
    ("clf", MultinomialNB()),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    cv_pipeline, full_df["Description_clean"], full_df["Category"],
    cv=skf, scoring="accuracy"
)

print("\n--- 5-Fold Stratified Cross-Validation ---")
print("Fold accuracies:", [f"{s:.3f}" for s in cv_scores])
print(f"Mean accuracy: {cv_scores.mean():.3f}  (Std: {cv_scores.std():.3f})")
print("Low std indicates the accuracy is stable across different train/test")
print("splits, not a result of one lucky split.")

# ---------------------------------------------------------------------
# 5. Save model + vectoriser
# ---------------------------------------------------------------------
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(model, "models/naive_bayes_model.pkl")
print("\nSaved: models/tfidf_vectorizer.pkl")
print("Saved: models/naive_bayes_model.pkl")