import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config import LAYERS_DIR, NUM_LAYERS, PROBE_N_EXPERIMENTS, PROBE_TEST_SIZE, PROBE_RANDOM_STATE


METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def probe_layer(X: np.ndarray, y: np.ndarray, n_experiments: int) -> dict:
    results = {m: [] for m in METRICS}
    for seed in range(n_experiments):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE + seed
        )
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        results["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        results["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        results["roc_auc"].append(roc_auc_score(y_test, y_prob))

    return {m: (np.mean(v), np.std(v)) for m, v in results.items()}


def main():
    rows = []
    for layer_idx in range(1, NUM_LAYERS + 1):
        path = os.path.join(LAYERS_DIR, f"layer{layer_idx}.csv")
        df = pd.read_csv(path)
        y = df["label"].values
        X = df.drop(columns=["label"]).values

        stats = probe_layer(X, y, PROBE_N_EXPERIMENTS)
        row = {"layer": layer_idx}
        for m, (mean, std) in stats.items():
            row[f"{m}_mean"] = round(mean, 4)
            row[f"{m}_std"] = round(std, 4)
        rows.append(row)
        print(
            f"Layer {layer_idx:2d}  "
            f"acc={row['accuracy_mean']:.4f}  "
            f"f1={row['f1_mean']:.4f}  "
            f"roc_auc={row['roc_auc_mean']:.4f}"
        )

    results_df = pd.DataFrame(rows)
    out_path = os.path.join("data", "probing_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    best = results_df.loc[results_df["roc_auc_mean"].idxmax()]
    print(f"Best layer by ROC-AUC: layer {int(best['layer'])} ({best['roc_auc_mean']:.4f})")


if __name__ == "__main__":
    main()
