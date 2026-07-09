"""
Model Comparison — Naive Bayes vs Logistic Regression vs Random Forest
==========================================================================

PURPOSE
-------
Trains three different classifier algorithms on the SAME TF-IDF features
and compares their performance side by side. This directly supports the
dissertation's "critical evaluation" learning outcome (COMP7039 Learning
Outcome 2 and 9) by demonstrating a fair, multi-algorithm comparison
rather than committing to a single technique without justification.

WHY THESE THREE MODELS
------------------------
1. NAIVE BAYES (baseline, see train_classifier.py): fast, interpretable,
   well-suited to word-frequency style features. Standard first choice
   in text classification literature.

2. LOGISTIC REGRESSION: a linear model that learns a weighted combination
   of TF-IDF features per category. Its coefficients are directly
   interpretable (which words push toward/away from a category), making
   it easy to explain and justify in a dissertation. Frequently used as
   a benchmark against Naive Bayes in NLP classification literature.

3. RANDOM FOREST: an ensemble of decision trees, representing a
   different algorithmic family (tree-based, non-linear) from the other
   two (probabilistic / linear). Including it demonstrates the comparison
   wasn't limited to superficially similar techniques, strengthening the
   "critical evaluation" argument in the Results/Discussion chapters.

WHY WE USE THE SAME TF-IDF FEATURES FOR ALL THREE
------------------------------------------------------
To make this a FAIR comparison, all three models are trained and
evaluated on IDENTICAL input features (same vectoriser, same
train/test split). This isolates the comparison to "which algorithm
performs better," rather than confounding it with "which algorithm got
better input features."

WHAT TO REPORT IN THE DISSERTATION
--------------------------------------
The printed comparison table (accuracy, macro-F1 per model) can be
copied directly into the Results chapter. The Discussion chapter should
explain WHY the winning model performs best (or why they perform
similarly) with reference to the nature of the task and data -- not just
report the numbers.

OUTPUTS
-------
models/logistic_regression_model.pkl
models/random_forest_model.pkl
(tfidf_vectorizer.pkl is reused from train_classifier.py -- not
re-saved here, since all three models share the identical vectoriser)
Printed: comparison table across all three models
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------------------------------------------------------------
# 1. Load train/test data
# ---------------------------------------------------------------------
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train_text = train_df["Description_clean"]
y_train = train_df["Category"]
X_test_text = test_df["Description_clean"]
y_test = test_df["Category"]

# ---------------------------------------------------------------------
# 2. Shared TF-IDF features (identical for all three models -- fair comparison)
# ---------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------------------------------------------------------------------
# 3. Define models
# ---------------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
trained_models = {}

# ---------------------------------------------------------------------
# 4. Train, predict, evaluate each model
# ---------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    results.append({"Model": name, "Accuracy": round(acc, 3), "Macro_F1": round(macro_f1, 3)})
    trained_models[name] = model

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------------------------------------
# 5. Comparison table
# ---------------------------------------------------------------------
comparison_df = pd.DataFrame(results).sort_values("Macro_F1", ascending=False).reset_index(drop=True)
print("\n--- Model Comparison Table (copy into Results chapter) ---")
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------------------
# 6. Save Logistic Regression and Random Forest models
# ---------------------------------------------------------------------
joblib.dump(trained_models["Logistic Regression"], "models/logistic_regression_model.pkl")
joblib.dump(trained_models["Random Forest"], "models/random_forest_model.pkl")
print("\nSaved: models/logistic_regression_model.pkl")
print("Saved: models/random_forest_model.pkl")