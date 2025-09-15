import logging
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
from commons.config import CONFLICT_LABELS, ID_TO_LABEL, setup_paths
from commons.data.load_and_preprocess_dataset import load_and_preprocess_dataset
from commons.metrics.classification_metrics import (
    compute_classification_metrics,
    print_classification_report,
)
from omegaconf import DictConfig, omegaconf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

# Setup paths and logging
setup_paths()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_function(examples, tokenizer, max_length=512):
    """
    Preprocess the data for training.

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer object
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs and labels
    """
    # Tokenize the combined text
    tokenized = tokenizer(
        examples["combined_text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Add labels
    tokenized["labels"] = examples["conflict_label"]

    return tokenized


def compute_metrics(p: EvalPrediction) -> dict:
    """
    Compute metrics for evaluation.

    Args:
        p: EvalPrediction object

    Returns:
        Dictionary with metrics
    """
    predictions, labels = p.predictions, p.label_ids

    # Get predicted class labels
    predictions = np.argmax(predictions, axis=1)

    # Compute basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


@hydra.main(config_name="default", config_path="configs", version_base="1.2")
def main(config: DictConfig):
    """
    Main function to train the conflict classification model.

    Args:
        config: Hydra configuration object
    """
    # Initialize wandb
    run = wandb.init(
        entity=config.logging.entity,
        project=config.logging.project,
        tags=config.logging.tags,
        settings=wandb.Settings(start_method="thread"),
    )

    # Update wandb config
    container = omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    run.config.update(container)

    logger.info("Loading and preprocessing dataset...")

    # Load datasets
    dataset_train, dataset_val, dataset_test = load_and_preprocess_dataset(
        data_path=config.data.data_path,
        test_ratio=config.data.test_ratio,
        val_ratio=config.data.val_ratio,
        max_length=config.data.max_length,
        random_state=config.global_seed,
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    dataset_train = dataset_train.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.data.max_length},
    )
    dataset_val = dataset_val.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.data.max_length},
    )
    dataset_test = dataset_test.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.data.max_length},
    )

    # Create label mapping
    id2label = ID_TO_LABEL
    label2id = {label: i for i, label in enumerate(CONFLICT_LABELS)}

    # Load model
    logger.info(f"Loading model: {config.model.model_name_or_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name_or_path,
        num_labels=config.model.num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.logging.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.epochs,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        push_to_hub=config.training.push_to_hub,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="f1",
        save_total_limit=3,
        report_to=["wandb"],
        lr_scheduler_type="linear",
        seed=config.global_seed,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=dataset_test, metric_key_prefix="test")
    logger.info(f"Test results: {test_results}")

    # Generate predictions
    logger.info("Generating predictions...")
    predictions, _, _ = trainer.predict(dataset_test)

    # Save predictions (simplified)
    run_name = run.name
    run_id = run.id
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save predictions as numpy array
    predictions_path = output_dir / f"predictions_{run_name}_{run_id}_{timestamp}.npy"
    np.save(predictions_path, predictions)

    # Save predictions as simple text file
    predicted_labels = np.argmax(predictions, axis=1)
    labels_path = output_dir / f"predicted_labels_{run_name}_{run_id}_{timestamp}.txt"
    with open(labels_path, "w") as f:
        for label in predicted_labels:
            f.write(f"{id2label[label]}\n")

    logger.info(f"Predictions saved to {predictions_path} and {labels_path}")

    # Print detailed classification report
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = [example["conflict_label"] for example in dataset_test]

    report = print_classification_report(np.array(true_labels), predicted_labels, CONFLICT_LABELS)
    logger.info(f"\nClassification Report:\n{report}")

    # Compute comprehensive metrics
    metrics = compute_classification_metrics(
        np.array(true_labels), predicted_labels, CONFLICT_LABELS
    )
    logger.info(f"Comprehensive metrics: {metrics}")

    # Log metrics to wandb
    wandb.log({"test_classification_report": report})
    wandb.log(metrics)

    wandb.finish()
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
