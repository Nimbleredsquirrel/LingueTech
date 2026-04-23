import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config import DATA_DIR, PLOTS_DIR, NUM_LAYERS, ALL_HIDDEN_PATH, LABELS_PATH

LAYERS_TO_PLOT = [1, 6, 11, 16, 21, 26, 32]


def plot_pca(X: np.ndarray, y: np.ndarray, layer_idx: int):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color, name in [(1, "steelblue", "correct"), (0, "salmon", "wrong")]:
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name, alpha=0.5, s=10)
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title(f"Layer {layer_idx} hidden states (PCA)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"pca_layer{layer_idx}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_roc_auc_curve():
    probing_path = os.path.join(DATA_DIR, "probing_results.csv")
    mm_path = os.path.join(DATA_DIR, "mm_probe_label_results.csv")

    fig, ax = plt.subplots(figsize=(9, 5))

    if os.path.exists(probing_path):
        df = pd.read_csv(probing_path)
        ax.plot(df["layer"], df["roc_auc_mean"], marker="o", label="Logistic Regression", color="steelblue")
        ax.fill_between(
            df["layer"],
            df["roc_auc_mean"] - df["roc_auc_std"],
            df["roc_auc_mean"] + df["roc_auc_std"],
            alpha=0.2, color="steelblue"
        )

    if os.path.exists(mm_path):
        df_mm = pd.read_csv(mm_path)
        ax.plot(df_mm["layer"], df_mm["mm_roc_auc_mean"], marker="s", label="Mass-Mean Probe", color="salmon")
        ax.plot(df_mm["layer"], df_mm["lda_roc_auc_mean"], marker="^", label="LDA Probe", color="seagreen")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("Layer")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Probe ROC-AUC across Llama-2-7b layers")
    ax.legend()
    ax.set_xticks(range(1, NUM_LAYERS + 1))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_auc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_hidden = np.load(ALL_HIDDEN_PATH)  # (NUM_LAYERS, n, HIDDEN_DIM)
    labels = np.load(LABELS_PATH)

    for layer_idx in LAYERS_TO_PLOT:
        plot_pca(all_hidden[layer_idx - 1], labels, layer_idx)

    plot_roc_auc_curve()


if __name__ == "__main__":
    main()
