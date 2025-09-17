import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import wandb
from commons.config import CONFLICT_LABELS, ID_TO_LABEL
from commons.data.load_and_preprocess_dataset import load_and_preprocess_dataset
from datasets import Dataset
from groq import Groq
from omegaconf import DictConfig, omegaconf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

baselines_path = Path(__file__).parent.parent.parent / "baselines"
sys.path.insert(0, str(baselines_path))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    "qwen/qwen3-32b"
    # ADD MORE MODELS HERE
]

DEFAULT_MODEL = "qwen/qwen3-32b"


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(self, conflict_labels: List[str] = None):
        self.conflict_labels = conflict_labels or CONFLICT_LABELS
        self.labels_str = ", ".join(self.conflict_labels)
        self.definitions = """Conflict type definitions:

1. `opposition` Conflicts: Contradictory findings about the same clinical entity
   Examples:
   - Normal vs abnormal findings of same body structure: Left breast: Unremarkable\
<> Left breast demonstrates persistent circumscribed masses
   - Negative vs positive statements: No cardiopulmonary disease <> Bibasilar atelectasis
   - Lab/vital sign interpretation: Low blood sugar at admission <> Patient\
 admitted with hyperglycemia
   - Opposite disorders: Hypernatremia <> Hyponatremia
   - Sex information opposites: Female patient <> Testis: Unremarkable

2. `anatomical` Conflicts: Contradictions regarding body structures and their\
 presence/absence
   Examples:
   - Absent vs present structures: Cholelithiasis <> The gallbladder is absent
   - History of removal vs present structure: Bilat mastectomy (2010) <>\
 Left breast: solid mass
   - Imaging vs clinical finding: Procedure: Chest XR <> Brain lesion
   - Laterality mismatch: Stable ductal carcinoma of left breast <> Right breast carcinoma

3. `value` Conflicts: Contradictory measurements, lab values, or quantitative findings
   Examples:
   - Condition vs measurement: Hypoglycemia <> Blood glucose 145
   - Conflicting lab measurements: 02/11/2022 WBC 8.0 <> 02/11/2022 WBC 5.5

4. `contraindication` Conflicts: Conflicts between allergies/contraindications\
 and treatments
   Examples:
   - Allergy vs prescribed medication: Allergic to acetaminophen <>\
 Home meds include Tylenol

5. `comparison` Conflicts: Contradictory comparative statements or temporal changes
   Examples:
   - Increased/decreased vs measurements: Ultrasound shows 3 cm lesion,\
 previously 4 cm, indicating increase

6. `descriptive` Conflicts: Contradictory descriptive statements about the same condition
   Examples:
   - Positive vs unlikely statements: Lungs: Pleural effusion unlikely <> Assessment:\
 Pleural effusion
   - Conflicting characteristics: Stable small pleural effusion <> Impression:\
 Small pleural effusion
   - Multiple vs single statements: Findings: 9 mm lesion right kidney <>\
 Assessment: Right renal lesions"""

    def _get_base_prompt(self) -> str:
        """Get the base prompt structure."""
        return f"""You are a medical expert tasked with classifying clinical text conflicts.

Your task is to analyze pairs of clinical documents and determine what type of \
conflict exists between them.

Available conflict types: {self.labels_str}

{self.definitions}

IMPORTANT: Respond with ONLY the conflict type name (one of: {self.labels_str}). \
Do not include any reasoning, explanation, or additional text."""

    def create_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a prompt for the given text and examples."""
        raise NotImplementedError


class ZeroShotPrompt(PromptTemplate):
    """Zero-shot prompt template."""

    def create_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a zero-shot prompt."""
        return f"""{self._get_base_prompt()}

Please analyze the following clinical text pair and classify the conflict type:

{text}

Conflict type:"""


class OneShotPrompt(PromptTemplate):
    """One-shot prompt template."""

    def create_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a one-shot prompt."""
        if not examples or len(examples) == 0:
            return ZeroShotPrompt(self.conflict_labels).create_prompt(text, examples)

        example = examples[0]
        return f"""{self._get_base_prompt()}

Here's an example:

Example:
{example['text']}
Conflict type: {example['label']}

Now analyze the following clinical text pair and respond with ONLY the conflict type:

{text}
Conflict type:"""


class FewShotPrompt(PromptTemplate):
    """Few-shot prompt template."""

    def create_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a few-shot prompt."""
        if not examples or len(examples) == 0:
            return ZeroShotPrompt(self.conflict_labels).create_prompt(text, examples)

        # Create examples section
        examples_text = ""
        for i, example in enumerate(examples, 1):
            examples_text += (
                f"Example {i}:\n{example['text']}\nConflict type: {example['label']}\n\n"
            )

        return f"""{self._get_base_prompt()}

Here are {len(examples)} examples, one for each conflict type:

{examples_text}

Now analyze the following clinical text pair and respond with ONLY the conflict type:

{text}
Conflict type:"""


def get_prompt_template(shot_type: str) -> PromptTemplate:
    """Get the appropriate prompt template based on shot type."""
    if shot_type == "zero":
        return ZeroShotPrompt()
    elif shot_type == "one":
        return OneShotPrompt()
    elif shot_type == "few":
        return FewShotPrompt()
    else:
        raise ValueError(f"Unknown shot type: {shot_type}. Must be 'zero', 'one', or 'few'.")


class GroqLLMClient:
    """Client for interacting with Groq API."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """Initialize the Groq client."""
        if api_key is None:
            api_key = self._get_api_key()

        self.client = Groq(api_key=api_key)
        self.model = model
        self.conflict_labels = CONFLICT_LABELS
        self.id_to_label = ID_TO_LABEL

    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set. "
                "Please set it with your Groq API key."
            )
        return api_key

    def classify_text(
        self,
        text: str,
        prompt_template: PromptTemplate,
        examples: List[Dict[str, Any]] = None,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Classify text using the LLM."""
        prompt = prompt_template.create_prompt(text, examples)

        for attempt in range(max_retries):
            try:
                # Use streaming to get the complete response
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    top_p=0.95,
                    stream=True,
                    stop=None,
                )

                # Collect the streaming response
                raw_response = ""
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        raw_response += chunk.choices[0].delta.content

                raw_response = raw_response.strip()

                # Check if we got any response
                if not raw_response:
                    raise ValueError("Empty response from LLM")
                logger.info(f"LLM raw response: {raw_response}")
                predicted_label = self._parse_response(raw_response)
                confidence = self._calculate_confidence(raw_response, predicted_label)
                logger.info(f"Parsed label: {predicted_label}, Confidence: {confidence}")

                return {
                    "predicted_label": predicted_label,
                    "raw_response": raw_response,
                    "confidence": confidence,
                    "success": True,
                    "error": None,
                }

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2**attempt))
                else:
                    return {
                        "predicted_label": "opposition",
                        "raw_response": "",
                        "confidence": 0.0,
                        "success": False,
                        "error": str(e),
                    }

    def _parse_response(self, response: str) -> str:
        """Parse the LLM response to extract the predicted label."""
        response = response.lower().strip()

        # First, try to find conflict type after "conflict type:" or at the start
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if "conflict type:" in line:
                # Extract text after "conflict type:"
                after_colon = line.split("conflict type:")[-1].strip()
                for label in self.conflict_labels:
                    if label.lower() in after_colon:
                        return label
            elif line and not line.startswith("<") and not line.startswith("think"):
                # Check if line starts with a conflict type
                for label in self.conflict_labels:
                    if line.startswith(label.lower()):
                        return label

        # Fallback: search for any conflict type in the response
        for label in self.conflict_labels:
            if label.lower() in response:
                return label

        # Last resort: check for partial matches
        for label in self.conflict_labels:
            if any(word in response for word in label.split()):
                return label

        logger.warning(f"Could not parse response: {response}")
        return "opposition"

    def _calculate_confidence(self, response: str, predicted_label: str) -> float:
        """Calculate confidence score based on response characteristics."""
        response = response.lower().strip()
        confidence = 0.5

        if len(response.split()) <= 3:
            confidence += 0.2

        if predicted_label.lower() in response:
            confidence += 0.3

        uncertainty_words = ["maybe", "possibly", "might", "could", "unclear", "uncertain"]
        if any(word in response for word in uncertainty_words):
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def batch_classify(
        self,
        texts: List[str],
        prompt_template: PromptTemplate,
        examples: List[Dict[str, Any]] = None,
        batch_size: int = 10,
        delay_between_batches: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Classify multiple texts in batches."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
            )

            batch_results = []
            for text in batch_texts:
                result = self.classify_text(text, prompt_template, examples)
                batch_results.append(result)

            results.extend(batch_results)

            if i + batch_size < len(texts):
                time.sleep(delay_between_batches)

        return results


class LLMEvaluator:
    """Evaluator for LLM baseline results."""

    def __init__(self, conflict_labels: List[str] = None):
        self.conflict_labels = conflict_labels or CONFLICT_LABELS
        self.label_to_id = {label: idx for idx, label in enumerate(self.conflict_labels)}

    def _convert_labels_to_ids(self, labels: List[str]) -> List[int]:
        """Convert string labels to integer IDs."""
        return [self.label_to_id[label] for label in labels]

    def _initialize_metrics_arrays(self) -> tuple:
        """Initialize arrays for per-class metrics."""
        length = len(self.conflict_labels)
        return (
            [0.0] * length,  # precision
            [0.0] * length,  # recall
            [0.0] * length,  # f1
            [0] * length,  # support
        )

    def _populate_per_class_metrics(
        self,
        unique_labels: List[int],
        precision_per_class: np.ndarray,
        recall_per_class: np.ndarray,
        f1_per_class: np.ndarray,
        support_per_class: np.ndarray,
    ) -> tuple:
        """Populate per-class metrics arrays."""
        all_precision, all_recall, all_f1, all_support = self._initialize_metrics_arrays()

        for i, label_id in enumerate(unique_labels):
            all_precision[label_id] = precision_per_class[i]
            all_recall[label_id] = recall_per_class[i]
            all_f1[label_id] = f1_per_class[i]
            all_support[label_id] = support_per_class[i]

        return all_precision, all_recall, all_f1, all_support

    def evaluate_predictions(
        self,
        true_labels: List[str],
        predicted_labels: List[str],
        confidence_scores: List[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate predictions against true labels."""
        true_ids = self._convert_labels_to_ids(true_labels)
        predicted_ids = self._convert_labels_to_ids(predicted_labels)

        accuracy = accuracy_score(true_ids, predicted_ids)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_ids, predicted_ids, average="weighted", zero_division=0
        )

        unique_labels = sorted(set(true_ids + predicted_ids))
        present_labels = [self.conflict_labels[i] for i in unique_labels]

        (
            precision_per_class,
            recall_per_class,
            f1_per_class,
            support_per_class,
        ) = precision_recall_fscore_support(
            true_ids, predicted_ids, labels=unique_labels, average=None, zero_division=0
        )

        class_report = classification_report(
            true_ids,
            predicted_ids,
            labels=unique_labels,
            target_names=present_labels,
            zero_division=0,
            output_dict=True,
        )

        # Convert NumPy types to Python native types in classification report
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, "item"):  # NumPy scalar
                return obj.item()
            else:
                return obj

        class_report = convert_numpy_types(class_report)

        cm = confusion_matrix(true_ids, predicted_ids)

        confidence_metrics = {}
        if confidence_scores:
            confidence_metrics = self._calculate_confidence_metrics(
                true_labels, predicted_labels, confidence_scores
            )

        all_precision, all_recall, all_f1, all_support = self._populate_per_class_metrics(
            unique_labels, precision_per_class, recall_per_class, f1_per_class, support_per_class
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision) if precision is not None else 0.0,
            "recall": float(recall) if recall is not None else 0.0,
            "f1": float(f1) if f1 is not None else 0.0,
            "support": int(support) if support is not None else 0,
            "per_class_metrics": {
                "precision": [float(x) if x is not None else 0.0 for x in all_precision],
                "recall": [float(x) if x is not None else 0.0 for x in all_recall],
                "f1": [float(x) if x is not None else 0.0 for x in all_f1],
                "support": [int(x) if x is not None else 0 for x in all_support],
            },
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "confidence_metrics": confidence_metrics,
            "unique_labels": [int(x) for x in unique_labels],
            "present_labels": present_labels,
        }

    def _separate_confidences_by_correctness(
        self, confidence_scores: List[float], correct_predictions: List[bool]
    ) -> tuple:
        """Separate confidence scores by prediction correctness."""
        correct_confidences = [
            c for c, correct in zip(confidence_scores, correct_predictions) if correct
        ]
        incorrect_confidences = [
            c for c, correct in zip(confidence_scores, correct_predictions) if not correct
        ]
        return correct_confidences, incorrect_confidences

    def _get_present_labels(self, metrics: Dict[str, Any]) -> List[str]:
        """Get present labels from metrics."""
        unique_labels = metrics.get("unique_labels", range(len(self.conflict_labels)))
        return [self.conflict_labels[i] for i in sorted(set(unique_labels))]

    def _calculate_confidence_metrics(
        self, true_labels: List[str], predicted_labels: List[str], confidence_scores: List[float]
    ) -> Dict[str, Any]:
        """Calculate confidence-based metrics."""
        correct_predictions = [t == p for t, p in zip(true_labels, predicted_labels)]
        correct_confidences, incorrect_confidences = self._separate_confidences_by_correctness(
            confidence_scores, correct_predictions
        )

        return {
            "avg_confidence_correct": float(np.mean(correct_confidences))
            if correct_confidences
            else 0.0,
            "avg_confidence_incorrect": float(np.mean(incorrect_confidences))
            if incorrect_confidences
            else 0.0,
            "confidence_std": float(np.std(confidence_scores)),
            "high_confidence_accuracy": self._calculate_high_confidence_accuracy(
                correct_predictions, confidence_scores, threshold=0.7
            ),
        }

    def _calculate_high_confidence_accuracy(
        self,
        correct_predictions: List[bool],
        confidence_scores: List[float],
        threshold: float = 0.7,
    ) -> float:
        """Calculate accuracy for high-confidence predictions."""
        high_conf_mask = [c >= threshold for c in confidence_scores]
        if not any(high_conf_mask):
            return 0.0

        high_conf_correct = [c for c, hc in zip(correct_predictions, high_conf_mask) if hc]
        return float(np.mean(high_conf_correct))

    def print_evaluation_report(self, metrics: Dict[str, Any]) -> str:
        """Print a formatted evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("LLM BASELINE EVALUATION REPORT")
        report.append("=" * 60)

        report.append("\nOverall Metrics:")
        report.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
        report.append(f"  Precision: {metrics['precision']:.4f}")
        report.append(f"  Recall:    {metrics['recall']:.4f}")
        report.append(f"  F1-Score:  {metrics['f1']:.4f}")

        report.append("\nPer-Class Metrics:")
        report.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        report.append("-" * 60)

        for i, label in enumerate(self.conflict_labels):
            precision = metrics["per_class_metrics"]["precision"][i]
            recall = metrics["per_class_metrics"]["recall"][i]
            f1 = metrics["per_class_metrics"]["f1"][i]
            support = metrics["per_class_metrics"]["support"][i]
            report.append(
                f"{label:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}"
            )

        if metrics["confidence_metrics"]:
            conf_metrics = metrics["confidence_metrics"]
            report.append("\nConfidence Metrics:")
            report.append(
                f"  Avg Confidence (Correct):   {conf_metrics['avg_confidence_correct']:.4f}"
            )
            report.append(
                f"  Avg Confidence (Incorrect): {conf_metrics['avg_confidence_incorrect']:.4f}"
            )
            report.append(f"  Confidence Std:             {conf_metrics['confidence_std']:.4f}")
            report.append(
                f"  High Confidence Accuracy:   {conf_metrics['high_confidence_accuracy']:.4f}"
            )

        report.append("\nConfusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        present_labels = self._get_present_labels(metrics)

        report.append(f"{'':<15} {'Predicted':<60}")
        report.append(f"{'Actual':<15} {' '.join([f'{label[:8]:<8}' for label in present_labels])}")
        report.append("-" * 80)

        for i, label in enumerate(present_labels):
            row = (
                f"{label[:14]:<15} "
                f"{' '.join([f'{cm[i,j]:<8}' for j in range(len(present_labels))])}"
            )
            report.append(row)

        report.append("=" * 60)

        return "\n".join(report)

    def save_evaluation_results(
        self, metrics: Dict[str, Any], output_path: str, additional_info: Dict[str, Any] = None
    ) -> None:
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "metrics": metrics,
            "conflict_labels": self.conflict_labels,
            "additional_info": additional_info or {},
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")


class LLMBaselineRunner:
    """Main runner for LLM baseline experiments."""

    def __init__(self, config: DictConfig):
        """Initialize the runner."""
        self.config = config
        self.llm_client = GroqLLMClient(api_key=config.api.get("api_key"), model=config.model.name)
        self.evaluator = LLMEvaluator(CONFLICT_LABELS)

        # Extract commonly used config values
        self.data_config = config.data
        self.experiment_config = config.experiment
        self.logging_config = config.logging

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        logger.info("Starting LLM baseline experiment...")

        run = wandb.init(
            entity=self.logging_config.entity,
            project=self.logging_config.project,
            tags=self.logging_config.tags + [self.experiment_config.shot_type],
            settings=wandb.Settings(start_method="thread"),
        )

        container = omegaconf.OmegaConf.to_container(
            self.config, resolve=True, throw_on_missing=True
        )
        run.config.update(container)

        try:
            logger.info("Loading and preprocessing dataset...")
            train_dataset, val_dataset, test_dataset = load_and_preprocess_dataset(
                data_path=self.data_config.data_path,
                test_ratio=self.data_config.test_ratio,
                val_ratio=self.data_config.val_ratio,
                max_length=self.data_config.max_length,
                random_state=self.config.global_seed,
            )

            prompt_template = get_prompt_template(self.experiment_config.shot_type)
            examples = self._prepare_examples(train_dataset, self.experiment_config.num_examples)

            logger.info(f"Running {self.experiment_config.shot_type}-shot evaluation...")
            results = self._evaluate_on_dataset(
                test_dataset, prompt_template, examples, self.experiment_config.batch_size
            )

            self._log_results(results, run)
            self._save_results(results, run)

            logger.info("Experiment completed successfully!")
            return results

        finally:
            wandb.finish()

    def _prepare_examples(self, train_dataset: Dataset, num_examples: int) -> List[Dict[str, Any]]:
        """Prepare examples for few-shot prompting."""
        if num_examples == 0:
            return []

        import random

        random.seed(self.config.global_seed)

        examples = []

        # For few-shot, ensure we have exactly one example from each category
        if self.experiment_config.shot_type == "few":
            # Group examples by conflict type
            examples_by_class = {}
            for example in train_dataset:
                label = example["conflict_type"]
                if label not in examples_by_class:
                    examples_by_class[label] = []
                examples_by_class[label].append(example)

            # Select one random example from each class
            for label in CONFLICT_LABELS:
                if label in examples_by_class and len(examples_by_class[label]) > 0:
                    selected_example = random.choice(examples_by_class[label])
                    examples.append({"text": selected_example["combined_text"], "label": label})

            # Shuffle the examples to avoid bias in order
            random.shuffle(examples)

            # For few-shot, we ignore num_examples and use all available categories
            logger.info(
                f"Prepared {len(examples)} examples (one per conflict type) "
                f"for {self.experiment_config.shot_type}-shot prompting"
            )
            return examples

        else:
            # For zero-shot and one-shot, use the original logic
            class_counts = {}
            for example in train_dataset:
                label = example["conflict_type"]
                if class_counts.get(label, 0) < num_examples // len(CONFLICT_LABELS) + 1:
                    examples.append({"text": example["combined_text"], "label": label})
                    class_counts[label] = class_counts.get(label, 0) + 1

                    if len(examples) >= num_examples:
                        break

            while len(examples) < num_examples and len(examples) < len(train_dataset):
                remaining = [ex for ex in train_dataset if ex not in [e["text"] for e in examples]]
                if remaining:
                    example = random.choice(remaining)
                    examples.append(
                        {"text": example["combined_text"], "label": example["conflict_type"]}
                    )

            logger.info(
                f"Prepared {len(examples)} examples "
                f"for {self.experiment_config.shot_type}-shot prompting"
            )
            return examples[:num_examples]

    def _extract_dataset_data(self, dataset: Dataset) -> tuple:
        """Extract texts and labels from dataset."""
        texts = [example["combined_text"] for example in dataset]
        true_labels = [example["conflict_type"] for example in dataset]
        return texts, true_labels

    def _build_results_dict(
        self,
        metrics: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        texts: List[str],
        true_labels: List[str],
        predicted_labels: List[str],
        confidence_scores: List[float],
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build results dictionary."""
        return {
            "metrics": metrics,
            "predictions": predictions,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "confidence_scores": confidence_scores,
            "examples_used": examples,
            "config": omegaconf.OmegaConf.to_container(self.config, resolve=True),
        }

    def _evaluate_on_dataset(
        self,
        dataset: Dataset,
        prompt_template: PromptTemplate,
        examples: List[Dict[str, Any]],
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """Evaluate on a dataset."""
        texts, true_labels = self._extract_dataset_data(dataset)

        logger.info(f"Classifying {len(texts)} texts...")
        predictions = self.llm_client.batch_classify(
            texts, prompt_template, examples, batch_size=batch_size
        )

        predicted_labels = [p["predicted_label"] for p in predictions]
        confidence_scores = [p["confidence"] for p in predictions]

        metrics = self.evaluator.evaluate_predictions(
            true_labels, predicted_labels, confidence_scores
        )

        return self._build_results_dict(
            metrics, predictions, texts, true_labels, predicted_labels, confidence_scores, examples
        )

    def _log_results(self, results: Dict[str, Any], run) -> None:
        """Log results to wandb."""
        metrics = results["metrics"]

        run.log(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

        for i, label in enumerate(CONFLICT_LABELS):
            run.log(
                {
                    f"precision_{label}": metrics["per_class_metrics"]["precision"][i],
                    f"recall_{label}": metrics["per_class_metrics"]["recall"][i],
                    f"f1_{label}": metrics["per_class_metrics"]["f1"][i],
                }
            )

        if metrics["confidence_metrics"]:
            conf_metrics = metrics["confidence_metrics"]
            run.log(
                {
                    "avg_confidence_correct": conf_metrics["avg_confidence_correct"],
                    "avg_confidence_incorrect": conf_metrics["avg_confidence_incorrect"],
                    "high_confidence_accuracy": conf_metrics["high_confidence_accuracy"],
                }
            )

    def _create_output_paths(self, run_name: str, timestamp: str) -> tuple:
        """Create output directory and file paths."""
        output_dir = Path("outputs") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        return (
            output_dir,
            output_dir / f"{run_name}_results.json",
            output_dir / f"{run_name}_predictions.json",
            output_dir / f"{run_name}_report.txt",
        )

    def _save_results(self, results: Dict[str, Any], run) -> None:
        """Save results to files."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = run.name or f"llm_baseline_{timestamp}"

        output_dir, results_path, predictions_path, report_path = self._create_output_paths(
            run_name, timestamp
        )

        self.evaluator.save_evaluation_results(
            results["metrics"],
            str(results_path),
            additional_info={
                "run_name": run_name,
                "run_id": run.id,
                "timestamp": timestamp,
                "config": results["config"],
            },
        )

        with open(predictions_path, "w") as f:
            json.dump(
                {
                    "predictions": results["predictions"],
                    "true_labels": results["true_labels"],
                    "predicted_labels": results["predicted_labels"],
                    "confidence_scores": results["confidence_scores"],
                },
                f,
                indent=2,
            )

        report = self.evaluator.print_evaluation_report(results["metrics"])
        logger.info(f"\n{report}")

        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Results saved to {output_dir}")


@hydra.main(config_name="default", config_path="../configs", version_base="1.2")
def main(config: DictConfig):
    """Main function to run LLM baseline experiments."""
    runner = LLMBaselineRunner(config)
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    main()
