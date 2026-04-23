import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import (
    MODEL_NAME, CACHE_DIR, HF_TOKEN, DATASET_PATH, LAYERS_DIR, DATA_DIR,
    NUM_LAYERS, HIDDEN_DIM, BATCH_SIZE,
    CHECKPOINT_EVERY, HIDDEN_CHECKPOINT_PATH, HIDDEN_PROGRESS_PATH,
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        output_hidden_states=True,
        load_in_8bit=True,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def extract_batch(texts: list[str], tokenizer, model) -> np.ndarray:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
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
    os.makedirs(LAYERS_DIR, exist_ok=True)
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

    print(f"Saving {NUM_LAYERS} layer CSVs...")
    col_names = [f"f{i}" for i in range(HIDDEN_DIM)]
    for layer_idx in range(NUM_LAYERS):
        layer_df = pd.DataFrame(all_hidden[layer_idx], columns=col_names)
        layer_df["label"] = labels
        layer_df.to_csv(os.path.join(LAYERS_DIR, f"layer{layer_idx + 1}.csv"), index=False)

    # cleanup checkpoint after successful completion
    for p in [HIDDEN_CHECKPOINT_PATH, HIDDEN_PROGRESS_PATH]:
        if os.path.exists(p):
            os.remove(p)

    print(f"Done. Saved {NUM_LAYERS} CSVs to {LAYERS_DIR}/")


if __name__ == "__main__":
    main()
