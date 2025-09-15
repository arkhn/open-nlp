# Conflict Detection and Classification Baselines

This directory contains baseline implementations for conflict detection and classification in
clinical documents.

## Overview

The baseline implements a text classification approach to detect and classify conflicts between
pairs of clinical documents into 6 categories:

1. **Opposition Conflicts** - Contradictory findings about the same clinical entity
2. **Anatomical Conflicts** - Contradictions regarding body structures and their presence/absence
3. **Value Conflicts** - Contradictory measurements, lab values, or quantitative findings
4. **Contraindication Conflicts** - Conflicts between allergies/contraindications and treatments
5. **Comparison Conflicts** - Temporal contradictions
6. **Descriptive Conflicts** - Statement contradictions

## Structure

```
baselines/
├── configs/
│   └── default.yaml          # Hydra configuration
├── commons/
│   ├── data/
│   │   └── load_and_preprocess_dataset.py
│   └── metrics/
│       └── classification_metrics.py
├── outputs/                  # Generated prediction files
├── train.py                  # Main training script
└── README.md
```

## Usage

### 1. Install Dependencies

```bash
pip install transformers datasets scikit-learn wandb hydra-core
```

### 2. Run Training

```bash
python train.py
```

### 3. Custom Configuration

```bash
python train.py model.model_name_or_path=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext training.epochs=15
```

## Configuration

The training can be configured via `configs/default.yaml`:

- **Model**: Uses BiomedBERT by default (can be changed to other models)
- **Data**: 80% train, 10% validation, 10% test split
- **Training**: 10 epochs, batch size 8, learning rate 2e-5
- **Logging**: Weights & Biases integration

## Data Format

The baseline expects a processed JSON file with the following structure:

- Each entry contains `data` with `doc_1` and `doc_2` text fields
- `annotations` contain conflict type information
- The loader extracts and creates:
  - `combined_text`: Concatenated text from both documents
  - `conflict_label`: Integer label (0-5) for conflict type
  - `conflict_type`: String label for conflict type
  - `doc1_id`, `doc2_id`: Document identifiers
  - `subject_id`: Patient identifier

## Output

The training generates:

- **Model checkpoints** in `outputs/`
- **Predictions** in `outputs/` (numpy arrays, text files)
- **Metrics** logged to Weights & Biases
- **Classification report** with per-class performance

## Metrics

The baseline computes:

- Accuracy
- Precision, Recall, F1 (macro and weighted)
- Per-class metrics
- Confusion matrix
- Confidence statistics

## Customization

To adapt for your data:

1. **Modify data loading** in `commons/data/load_and_preprocess_dataset.py`
2. **Adjust preprocessing** in the `preprocess_function`
3. **Update conflict types** in the configuration
4. **Change model** via Hydra overrides

## Example

```bash
# Train with different model
python train.py model.model_name_or_path=bert-base-uncased

# Train with different hyperparameters
python train.py training.epochs=20 training.learning_rate=1e-5

# Train with different data split
python train.py data.test_ratio=0.15 data.val_ratio=0.15
```
