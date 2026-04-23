# LingueTech: Probing Reasoning in LLMs

Probes Llama-2-7b hidden states to detect correct vs. wrong reasoning steps using logistic regression, mass-mean probing, LDA, and INSIDE EigenScore on the PRM800K dataset.

## Project Structure

```
LingueTech/
├── config.py                  # paths, model name, hyperparameters, concepts
├── prepare_dataset.py         # PRM800K JSONL -> data/dataset.parquet
├── extract_hidden_states.py   # Llama-2-7b hidden states -> layers/layer{i}.csv
├── probing.py                 # logistic regression probing per layer
├── mass_mean_probe.py         # mass-mean and LDA probing (Geometry of Truth)
├── pca_viz.py                 # PCA scatter plots and ROC-AUC curve
├── eigenscore.py              # INSIDE EigenScore (hallucination detection)
├── run_kaggle.ipynb           # Kaggle orchestration notebook
└── requirements.txt
```

## Methods

**Logistic Regression Probing** — trains a linear classifier on each layer's last-token hidden state. Higher ROC-AUC in deeper layers indicates those layers encode step correctness more linearly.

**Mass-Mean Probing** (Marks & Tegmark, COLM 2024) — uses the difference of class centroids as the probe direction: `theta = mu_plus - mu_minus`. Compared with LDA (Fisher's Linear Discriminant). Run for 8 concepts including correctness, certainty, hedging, negation, and error acknowledgment.

**INSIDE EigenScore** (Chen et al. 2024) — generates N continuations per prompt, extracts hidden states, computes eigenvalue entropy of the cosine similarity matrix. Low entropy = consistent responses = model is confident = likely correct step.

## Running on Kaggle

Open `run_kaggle.ipynb` on Kaggle (GPU T4 recommended). The notebook:
1. Checks GPU availability
2. Installs dependencies
3. Reads your HuggingFace token from Kaggle Secrets (`HF_TOKEN`)
4. Clones this repo and downloads PRM800K via git lfs
5. Runs the full pipeline end-to-end

## Running locally (requires GPU + HF token)

```bash
pip install -r requirements.txt

# get PRM800K
git lfs install
git clone https://github.com/openai/prm800k.git
mkdir -p data
cp prm800k/data/phase2/train.jsonl data/phase2_train.jsonl

# run pipeline
export HF_TOKEN=<your_token>
python prepare_dataset.py
python extract_hidden_states.py   # ~30-60 min on T4, checkpoints every 50 batches
python probing.py
python mass_mean_probe.py --all   # all 8 concepts
python pca_viz.py
python eigenscore.py              # ~2-4 hours on T4
```

## Concepts probed

| Concept | Description |
|---------|-------------|
| `label` | Step correctness (PRM800K ground truth) |
| `has_equation` | Contains `=` sign |
| `is_long_step` | Step length above median |
| `has_conclusion_word` | Concluding language (therefore/thus/hence/so) |
| `has_certainty` | Certainty markers (clearly/obviously/must/certainly) |
| `has_hedging` | Uncertainty markers (maybe/might/approximately/roughly) |
| `has_negation` | Negation present (not/no/never/neither) |
| `has_error_word` | Error acknowledgment (mistake/wrong/incorrect/invalid) |
