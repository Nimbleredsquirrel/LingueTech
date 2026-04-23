# how to get the data:
#   git lfs install
#   git clone https://github.com/openai/prm800k.git
#   cp prm800k/data/phase2/train.jsonl data/phase2_train.jsonl

import json
import os
import re
import pandas as pd
from config import DATA_DIR, PRM800K_PATH, DATASET_PATH, MAX_SAMPLES

CONCLUSION_PATTERN = re.compile(r"\b(?:therefore|thus|hence|so)\b", re.IGNORECASE)
CERTAINTY_PATTERN = re.compile(r"\b(?:clearly|obviously|exactly|precisely|must|certainly|definitely)\b", re.IGNORECASE)
HEDGING_PATTERN = re.compile(r"\b(?:maybe|might|possibly|approximately|roughly|could|perhaps|about)\b", re.IGNORECASE)
NEGATION_PATTERN = re.compile(r"\b(?:not|no|never|neither|nor|cannot|can't|isn't|doesn't)\b", re.IGNORECASE)
ERROR_PATTERN = re.compile(r"\b(?:mistake|error|wrong|incorrect|invalid|false|contradiction)\b", re.IGNORECASE)


def load_prm800k(path: str) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            problem = obj["question"]["problem"]
            for step in obj["label"]["steps"]:
                for completion in step["completions"]:
                    rating = completion.get("rating")
                    # skip neutral and unlabeled steps
                    if rating is None or rating == 0:
                        continue
                    step_text = completion["text"]
                    records.append({
                        "text": f"Problem: {problem}\nStep: {step_text}",
                        "prompt": f"Problem: {problem}\nStep:",
                        "label": 1 if rating == 1 else 0,
                        "step_text": step_text,
                    })
    return records


def derive_concepts(df: pd.DataFrame) -> pd.DataFrame:
    df["has_equation"] = df["step_text"].str.contains("=", regex=False).astype(int)
    median_len = df["step_text"].str.len().median()
    df["is_long_step"] = (df["step_text"].str.len() > median_len).astype(int)
    df["has_conclusion_word"] = df["step_text"].str.contains(CONCLUSION_PATTERN).astype(int)
    df["has_certainty"] = df["step_text"].str.contains(CERTAINTY_PATTERN).astype(int)
    df["has_hedging"] = df["step_text"].str.contains(HEDGING_PATTERN).astype(int)
    df["has_negation"] = df["step_text"].str.contains(NEGATION_PATTERN).astype(int)
    df["has_error_word"] = df["step_text"].str.contains(ERROR_PATTERN).astype(int)
    df = df.drop(columns=["step_text"])
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    records = load_prm800k(PRM800K_PATH)
    df = pd.DataFrame(records)
    df = derive_concepts(df)

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)

    df.to_parquet(DATASET_PATH, index=False)
    print(f"Saved {len(df)} rows to {DATASET_PATH}")
    print(f"Labels: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
