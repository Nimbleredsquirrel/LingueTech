import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from config import (
    MODEL_NAME, CACHE_DIR, HF_TOKEN, DATASET_PATH, HIDDEN_DIM,
    INSIDE_N_RESPONSES, INSIDE_TEMPERATURE, INSIDE_MAX_NEW_TOKENS,
    INSIDE_LAYER_IDX, INSIDE_RESPONSES_PATH, INSIDE_RESULTS_PATH, DATA_DIR
)


def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def generate_response_hidden_states(
    prompt: str,
    model,
    tokenizer,
    n_responses: int,
    layer_idx: int,
    temperature: float,
    max_new_tokens: int,
) -> np.ndarray:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(model.device)

    hidden_states_list = []
    for _ in range(n_responses):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # outputs.hidden_states: tuple[num_new_tokens] of tuple[num_layers] of (batch, seq, hidden)
        if not outputs.hidden_states:
            hidden_states_list.append(np.zeros(HIDDEN_DIM, dtype=np.float32))
            continue

        last_step_layers = outputs.hidden_states[-1]
        hs = last_step_layers[layer_idx][0, -1, :].float().cpu().numpy()
        hidden_states_list.append(hs)

    return np.stack(hidden_states_list, axis=0)  # (n_responses, hidden_dim)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    normalized = matrix / norms
    return normalized @ normalized.T


def eigenscore(response_hs: np.ndarray) -> float:
    # low entropy = consistent responses = model confident = likely correct step
    S = cosine_similarity_matrix(response_hs)
    eigenvalues = np.linalg.eigvalsh(S)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    if total < 1e-10:
        return 0.0
    p = eigenvalues / total
    p = p[p > 1e-10]
    return float(-np.sum(p * np.log(p)))


def generate_all_responses(df: pd.DataFrame, model, tokenizer) -> np.ndarray:
    n = len(df)
    all_hs = np.zeros((n, INSIDE_N_RESPONSES, HIDDEN_DIM), dtype=np.float32)
    for i, prompt in enumerate(df["prompt"]):
        all_hs[i] = generate_response_hidden_states(
            prompt, model, tokenizer,
            n_responses=INSIDE_N_RESPONSES,
            layer_idx=INSIDE_LAYER_IDX,
            temperature=INSIDE_TEMPERATURE,
            max_new_tokens=INSIDE_MAX_NEW_TOKENS,
        )
        if i % 100 == 0:
            print(f"  {i}/{n}")
    return all_hs


def compute_and_save_scores(all_hs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    scores = np.array([eigenscore(all_hs[i]) for i in range(all_hs.shape[0])])

    results_df = pd.DataFrame({"label": labels, "eigenscore": scores})
    results_df.to_csv(INSIDE_RESULTS_PATH, index=False)

    # negate: lower eigenscore = more consistent = correct
    auc = roc_auc_score(labels, -scores)
    print(f"ROC-AUC (lower eigenscore predicts correct): {auc:.4f}")
    print(f"Mean eigenscore  correct: {scores[labels == 1].mean():.4f}")
    print(f"Mean eigenscore  wrong:   {scores[labels == 0].mean():.4f}")
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=f"Skip generation, load cached {INSIDE_RESPONSES_PATH}",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_parquet(DATASET_PATH)
    labels = df["label"].values

    if args.eval_only:
        if not os.path.exists(INSIDE_RESPONSES_PATH):
            raise FileNotFoundError(
                f"{INSIDE_RESPONSES_PATH} not found. Run without --eval-only first."
            )
        all_hs = np.load(INSIDE_RESPONSES_PATH)
    else:
        tokenizer, model = load_model()
        print(f"Generating {INSIDE_N_RESPONSES} responses per sample for {len(df)} samples...")
        all_hs = generate_all_responses(df, model, tokenizer)
        np.save(INSIDE_RESPONSES_PATH, all_hs)

    compute_and_save_scores(all_hs, labels)


if __name__ == "__main__":
    main()
