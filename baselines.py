"""Reference classifiers on the same split as classify.py: majority class and TF-IDF + logistic regression.

    uv run python baselines.py
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from classify import load_splits


def main():
    train, val, test = load_splits()
    results = {}
    for label in ["sentiment", "emotion"]:
        majority = train[label].mode()[0]
        pred = [majority] * len(test)
        results[f"{label}_majority"] = {
            "test_acc": accuracy_score(test[label], pred),
            "test_macro_f1": f1_score(test[label], pred, average="macro"),
        }
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        clf = LogisticRegression(C=10, class_weight="balanced", max_iter=2000)
        clf.fit(vec.fit_transform(train.text), train[label])
        pred = clf.predict(vec.transform(test.text))
        results[f"{label}_tfidf_lr"] = {
            "test_acc": accuracy_score(test[label], pred),
            "test_macro_f1": f1_score(test[label], pred, average="macro"),
        }
    out = Path("runs") / "cls"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        print(f"{k:22s} acc {v['test_acc']:.4f} | macro-F1 {v['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
