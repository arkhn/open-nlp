from datasets import load_dataset


def load_preprocess_data(dataset_name, input_column, output_column, tokenizer):
    """
    Load the dataset and preprocess it for training

    Args:
        dataset_name (str): Name of the dataset to load
        input_column (str): Name of the column containing the input
        output_column (str): Name of the column containing the output
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for tokenization

    Returns:
        (datasets.Dataset, datasets.Dataset): Train and validation datasets
    """
    dataset = load_dataset(dataset_name)

    def preprocess_function(examples):
        inputs = examples[input_column]
        outputs = examples[output_column]
        tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True)
        tokenized_outputs = tokenizer(outputs, padding="max_length", truncation=True)
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_outputs["input_ids"],
        }

    tokenized_datasets = dataset.map(preprocess_function, batched=True)["train"]

    # split train and validation with 0.9 and 0.1 ratio
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["test"]

    return train_dataset, val_dataset
