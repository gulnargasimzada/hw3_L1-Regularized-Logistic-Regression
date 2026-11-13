import numpy as np
from pathlib import Path
import json
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path("data")

with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.9
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)
clf = LogisticRegression(
    l1_ratio=1,
    solver="liblinear",
    max_iter=2000
)

clf.fit(X_train, y_train)

#test set predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

###ROC AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 3))
### Model sparsity diagnostic
coefs = clf.coef_[0]
n_nonzero = np.count_nonzero(coefs)
n_total = len(coefs)
print(f"\nNon-zero coefficients: {n_nonzero} / {n_total} "
      f"({100 * n_nonzero / n_total:.1f}%)")
### Top words (interpretability)
feature_names = vectorizer.get_feature_names_out()
order = np.argsort(coefs)   # indices sorted from most negative to most positive

# Top 15 positive-weight words -> CORE (1)
top_pos = [i for i in order[::-1] if coefs[i] > 0][:15]
print("\nTop 15 positive-weight words (CORE = 1):")
for i in top_pos:
    print(f"  {feature_names[i]:<20} {coefs[i]:.4f}")

# Top 15 negative-weight words -> NEG (0); only strictly < 0
top_neg = [i for i in order if coefs[i] < 0][:15]
print(f"\nTop 15 negative-weight words (NEG = 0) [{len(top_neg)} found]:")
for i in top_neg:
    print(f"  {feature_names[i]:<20} {coefs[i]:.4f}")

### Misclassified examples (for label-noise analysis)
fp = [i for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 1]
fn = [i for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 0]

# Sort by model confidence (most confident mistakes first)
fp.sort(key=lambda i: y_prob[i], reverse=True)   # highest P(CORE) first
fn.sort(key=lambda i: y_prob[i])                 # lowest P(CORE) first

print(f"\nFalse positives (label NEG, predicted CORE): {len(fp)}")
print(f"False negatives (label CORE, predicted NEG): {len(fn)}")

N_SHOW = 5
for name, idxs in [("FALSE POSITIVES", fp), ("FALSE NEGATIVES", fn)]:
    print(f"\n===== {name} (top {N_SHOW}) =====")
    for i in idxs[:N_SHOW]:
        text = " ".join(X_test_texts[i].split())
        print(f"\n[test idx {i}] label={y_test[i]} pred={y_pred[i]} P(CORE)={y_prob[i]:.3f}")
        print(text)

# Save ALL misclassified chunks to a CSV file
Path("outputs").mkdir(exist_ok=True)
with open(Path("outputs") / "hw03_l1_misclassified.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["test_idx", "error_type", "label", "pred", "p_core", "text"])
    for etype, idxs in [("FP", fp), ("FN", fn)]:
        for i in idxs:
            writer.writerow([i, etype, y_test[i], y_pred[i], round(y_prob[i], 4), X_test_texts[i]])

print("\nAll misclassified chunks saved to outputs/hw03_l1_misclassified.csv")