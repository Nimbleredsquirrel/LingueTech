import argparse
import os
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from config import (
    LAYERS_DIR, NUM_LAYERS, DATASET_PATH, DATA_DIR, PLOTS_DIR,
    PROBE_TEST_SIZE, PROBE_RANDOM_STATE, PROBE_N_EXPERIMENTS, CONCEPTS
)


def mass_mean_direction(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    mu_plus = X_train[y_train == 1].mean(axis=0)
    mu_minus = X_train[y_train == 0].mean(axis=0)
    return mu_plus - mu_minus


def mass_mean_score(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ theta)


def lda_score(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return lda.predict_proba(X_test)[:, 1]


def evaluate_layer(X: np.ndarray, y: np.ndarray) -> dict:
    mm_aucs, lda_aucs = [], []
    for seed in range(PROBE_N_EXPERIMENTS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE + seed
        )
        theta = mass_mean_direction(X_train, y_train)
        mm_aucs.append(roc_auc_score(y_test, mass_mean_score(theta, X_test)))
        lda_aucs.append(roc_auc_score(y_test, lda_score(X_train, y_train, X_test)))

    return {
        "mm_roc_auc_mean": float(np.mean(mm_aucs)),
        "mm_roc_auc_std": float(np.std(mm_aucs)),
        "lda_roc_auc_mean": float(np.mean(lda_aucs)),
        "lda_roc_auc_std": float(np.std(lda_aucs)),
    }


def probe_concept(concept: str, concept_labels: np.ndarray) -> pd.DataFrame:
    if len(np.unique(concept_labels)) < 2:
        raise ValueError(f"Concept '{concept}' has only one class in the data.")

    rows = []
    for layer_idx in range(1, NUM_LAYERS + 1):
        df = pd.read_csv(os.path.join(LAYERS_DIR, f"layer{layer_idx}.csv"))
        X = df.drop(columns=["label"]).values
        stats = evaluate_layer(X, concept_labels)
        rows.append({"layer": layer_idx, **stats})
        print(
            f"  Layer {layer_idx:2d}  MM={rows[-1]['mm_roc_auc_mean']:.4f}  "
            f"LDA={rows[-1]['lda_roc_auc_mean']:.4f}"
        )

    return pd.DataFrame(rows)


def plot_results(results: dict[str, pd.DataFrame]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    colors = ["steelblue", "salmon", "seagreen", "orchid", "sandybrown"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, method, title in zip(axes, ["mm", "lda"], ["Mass-Mean", "LDA"]):
        for (concept, df), color in zip(results.items(), colors):
            desc = CONCEPTS.get(concept, concept)
            col = f"{method}_roc_auc_mean"
            std_col = f"{method}_roc_auc_std"
            ax.plot(df["layer"], df[col], marker="o", label=desc, color=color, markersize=4)
            ax.fill_between(
                df["layer"],
                df[col] - df[std_col],
                df[col] + df[std_col],
                alpha=0.15, color=color,
            )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random")
        ax.set_xlabel("Layer")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{title} probe: concept ROC-AUC per layer")
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, NUM_LAYERS + 1, 2))

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "mass_mean_concepts.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept",
        nargs="+",
        default=["label"],
        help=f"Concept column(s) from dataset.parquet. Available: {list(CONCEPTS.keys())}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all concepts defined in config.CONCEPTS",
    )
    args = parser.parse_args()

    concepts_to_run = list(CONCEPTS.keys()) if args.all else args.concept

    dataset_df = pd.read_parquet(DATASET_PATH)
    missing = [c for c in concepts_to_run if c not in dataset_df.columns]
    if missing:
        raise ValueError(
            f"Concepts not found in dataset.parquet: {missing}. "
            f"Available: {list(dataset_df.columns)}"
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    all_results = {}

    for concept in concepts_to_run:
        print(f"\nProbing concept: '{concept}'")
        labels = dataset_df[concept].values
        results_df = probe_concept(concept, labels)
        out_path = os.path.join(DATA_DIR, f"mm_probe_{concept}_results.csv")
        results_df.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")
        all_results[concept] = results_df

        best_mm = results_df.loc[results_df["mm_roc_auc_mean"].idxmax()]
        best_lda = results_df.loc[results_df["lda_roc_auc_mean"].idxmax()]
        print(
            f"  Best MM:  layer {int(best_mm['layer'])} ({best_mm['mm_roc_auc_mean']:.4f})\n"
            f"  Best LDA: layer {int(best_lda['layer'])} ({best_lda['lda_roc_auc_mean']:.4f})"
        )

    if len(all_results) > 1:
        plot_results(all_results)


if __name__ == "__main__":
    main()
