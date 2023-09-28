from datasets import load_dataset


def load_and_preprocess_dataset():
    """Load and preprocess the dataset.

    Args:
        test_ratio (float): Ratio of the test set.
        seed (int): Seed for the random number generator.

    Returns:
        Tuple: Tuple with the train, validation, test and submission datasets.
    """

    dataset = load_dataset("bio-datasets/dft23-full")

    dataset_sub = dataset["test"]
    dataset_train = dataset["train"]

    return dataset_train, dataset["validation"], dataset_sub, dataset_sub
