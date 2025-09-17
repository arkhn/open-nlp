# LLM Baselines for Clinical Text Conflict Classification

This package provides zero-shot, one-shot, and few-shot prompting baselines for clinical text
conflict classification using various LLM models via Groq API.

## Installation

1. Install dependencies:

```bash
cd lib/conflicts/llm_baselines
pip install -e .
```

2. Set up your Groq API key:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

## Usage

### Basic Usage

Run a zero-shot baseline:

```bash
python run_baseline.py --config-name=zero_shot
```

Run a one-shot baseline:

```bash
python run_baseline.py --config-name=one_shot
```

Run a few-shot baseline:

```bash
python run_baseline.py --config-name=few_shot
```

### Advanced Usage

Run with a specific model:

```bash
python run_baseline.py --config-name=qwen_models model.name=qwen/qwen2.5-72b-instruct
```

Run with custom parameters:

```bash
python run_baseline.py \
    --config-name=few_shot \
    experiment.num_examples=10 \
    experiment.batch_size=5 \
    model.name=qwen/qwen2.5-14b-instruct
```

### Configuration

The package uses Hydra for configuration management. Key configuration options:

- `experiment.shot_type`: "zero", "one", or "few"
- `experiment.num_examples`: Number of examples for few-shot prompting
- `experiment.batch_size`: Batch size for processing
- `model.name`: Model name from Groq API
- `data.data_path`: Path to the conflict dataset

## Project Structure

This package uses a simplified single-file design:

```
llm_baselines/
├── llm_baseline.py    # All functionality in one consolidated module
├── __init__.py        # Package exports
└── configs/           # Configuration files
```

All classes and functions are available from the main module:

- `LLMBaselineRunner` - Main experiment runner
- `GroqLLMClient` - Groq API client
- `LLMEvaluator` - Evaluation metrics
- `ZeroShotPrompt`, `OneShotPrompt`, `FewShotPrompt` - Prompt templates
- `get_prompt_template()` - Factory function for prompts

## Requirements

- Python 3.11+
- Groq API key
- See `pyproject.toml` for full dependency list
