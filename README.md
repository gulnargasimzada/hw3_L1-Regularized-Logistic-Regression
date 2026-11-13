# HW3: L1-Regularized Logistic Regression (IDS 570: Text as Data)

Re-runs the Week 09 TF–IDF + logistic regression classifier (CORE vs NEG merchant chunks)
and compares the default **L2** penalty with **L1** (Lasso) on the same train/test split.
The goal is to compare both **performance** and **interpretability**, and to check how
each penalty responds to errors in the weakly supervised labels.

## Corpus

- 571 plain-text (`.txt`) documents in Early Modern English, with historical spelling
  (e.g. *marchant*, *booke*).
- After chunking: **534,933** chunks (CORE 9,774 · NEG 524,690 · MAYBE 469).
- After balancing and splitting: **15,638** train / **3,910** test chunks.

## Workflow

**1. Chunking and weak labeling (`step7_label_chunks.py`)**
Each document is split into sentences (NLTK) and the sentences are grouped into chunks of
about 120 words. Chunks with fewer than 5 or more than 200 tokens are dropped. Each chunk
then receives a rule-based label from three keyword tiers:
- **CORE (1):** contains a Tier A word (*merchant, merchants, marchant, marchants*) or a
  Tier B word (*factor, chapman, adventurer(s), venturer(s), staple, staplers, trade, purser*)
- **MAYBE (2):** contains only Tier C words (related occupations such as *clothier, goldsmith, banker*)
- **NEG (0):** contains none of them

These labels come from a keyword rule, not from human reading, so some of them are wrong
(weak supervision).

**2. Balanced train/test split (`step9_prepare_datasets.py`)**
CORE is only about 1.8% of all chunks, so NEG is randomly downsampled to the size of CORE
(`random.seed(42)`). The balanced set is shuffled and split 80/20 into
`train_core_vs_neg.json` and `test_core_vs_neg.json`. MAYBE chunks are held out.

**3. L2 baseline (`hw03_l2_baseline.py`)**
TF–IDF features (`lowercase=True, min_df=5, max_df=0.9`) are fitted on the training text
only and applied to the test text. A logistic regression with the default L2 penalty is
trained and evaluated.

**4. L1 model (`hw03_l1_model.py`)**
The same pipeline with one change: L1 penalty (`l1_ratio=1`, `solver="liblinear"`,
`max_iter=2000`). Data, TF–IDF settings and the test set stay identical, so any
difference comes from the penalty.

**5. Evaluation and interpretation (both scripts)**
Each script prints:
- confusion matrix, classification report (precision / recall / F1) and ROC AUC
- number of non-zero coefficients (sparsity)
- top 15 positive-weight words (CORE) and top 15 negative-weight words (NEG);
  only coefficients strictly above / below zero are listed
- the 5 most confident false positives and false negatives

All misclassified chunks are saved to a CSV file for the label-noise analysis.

## Files

| File | Purpose |
|---|---|
| `step7_label_chunks.py` | Chunking and weak labeling |
| `step9_prepare_datasets.py` | Balanced 80/20 train/test split |
| `hw03_l2_baseline.py` | Logistic regression with L2 (default) |
| `hw03_l1_model.py` | Logistic regression with L1 |
| `hw03_l2_misclassified.csv`, `hw03_l1_misclassified.csv` | All misclassified test chunks |

## How to run

Environment: Python 3.12, scikit-learn 1.9.1, nltk 3.9.1, gensim 4.4.0.

```bash
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install scikit-learn gensim tqdm "nltk==3.9.1"
```

1. Place the corpus `.txt` files in a `texts/` folder.
2. Create the labeled data and the train/test split:
   ```bash
   python step7_label_chunks.py
   python step9_prepare_datasets.py
   ```
3. Run the two models:
   ```bash
   python hw03_l2_baseline.py
   python hw03_l1_model.py
   ```

`texts/`, `data/` and `.venv/` are not included in the repo (see `.gitignore`).

**Note:** nltk 3.10+ adds a 5-second regex time limit that fails on the largest
corpus files, so nltk 3.9.1 is used.

## Results (test set: 3,910 chunks)

| | L2 | L1 |
|---|---|---|
| CORE F1 | 0.95 | 1.00 |
| ROC AUC | 0.987 | 1.0 |
| Non-zero coefficients | 16,198 / 16,198 | 28 / 16,198 |
| Misclassified (FP / FN) | 75 / 131 | 0 / 6 |