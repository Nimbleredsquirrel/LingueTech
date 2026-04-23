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
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        output_hidden_states=True,
        quantization_config=bnb_config,
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
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # last non-padding token hidden state for each layer
    last_token_idx = inputs["attention_mask"].sum(dim=1) - 1
    hidden_states = outputs.hidden_states  # tuple of NUM_LAYERS tensors

    batch_size = len(texts)
    result = np.zeros((NUM_LAYERS, batch_size, HIDDEN_DIM), dtype=np.float32)
    for layer_idx, layer_hs in enumerate(hidden_states):
        for sample_idx in range(batch_size):
            tok_idx = last_token_idx[sample_idx].item()
            result[layer_idx, sample_idx] = (
                layer_hs[sample_idx, tok_idx].float().cpu().numpy()
            )
    return result


def save_checkpoint(all_hidden: np.ndarray, done: int):
    np.save(HIDDEN_CHECKPOINT_PATH, all_hidden)
    with open(HIDDEN_PROGRESS_PATH, "w") as f:
        f.write(str(done))


def load_checkpoint(n: int) -> tuple[np.ndarray, int]:
    if os.path.exists(HIDDEN_CHECKPOINT_PATH) and os.path.exists(HIDDEN_PROGRESS_PATH):
        all_hidden = np.load(HIDDEN_CHECKPOINT_PATH)
        with open(HIDDEN_PROGRESS_PATH) as f:
            done = int(f.read().strip())
        if all_hidden.shape == (NUM_LAYERS, n, HIDDEN_DIM):
            print(f"Resuming from checkpoint ({done}/{n} samples done)")
            return all_hidden, done
    return np.zeros((NUM_LAYERS, n, HIDDEN_DIM), dtype=np.float32), 0


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_parquet(DATASET_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    n = len(texts)

    tokenizer, model = load_model()

    all_hidden, start_from = load_checkpoint(n)

    batch_num = 0
    for start in range(start_from, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_hidden = extract_batch(texts[start:end], tokenizer, model)
        all_hidden[:, start:end, :] = batch_hidden
        batch_num += 1
        if batch_num % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_hidden, end)
            print(f"  {end}/{n} (checkpoint saved)")
        elif batch_num % 10 == 0:
            print(f"  {end}/{n}")

    print("Saving hidden states...")
    np.save(ALL_HIDDEN_PATH, all_hidden)
    np.save(LABELS_PATH, np.array(labels))

    for p in [HIDDEN_CHECKPOINT_PATH, HIDDEN_PROGRESS_PATH]:
        if os.path.exists(p):
            os.remove(p)

    size_gb = all_hidden.nbytes / 1e9
    print(f"Done. Saved {ALL_HIDDEN_PATH} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
