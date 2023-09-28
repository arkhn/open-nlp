import requests
from huggingface_hub import HfApi

hf_api = HfApi()


def get_bigbio_datasets():
    bigbio_datasets = hf_api.list_datasets(author="bigbio")
    bigbio_names = [dataset.id.split("/")[-1] for dataset in bigbio_datasets]
    return bigbio_names


def get_ner_bigbio_datasets():
    bigbio_names = get_bigbio_datasets()
    base_url = "https://huggingface.co/datasets/bigbio/{}/raw/main/README.md"

    ner_bigbio_datasets_names = []

    # Loop through each name and check if the phrase exists in the corresponding README.md file
    for name in bigbio_names:
        url = base_url.format(name)
        response = requests.get(url)
        if "NAMED_ENTITY_RECOGNITION" in response.text:
            ner_bigbio_datasets_names.append(name)

    # Print the list of names that contain "NAMED_ENTITY_RECOGNITION"
    return ner_bigbio_datasets_names


if __name__ == "__main__":
    names = get_ner_bigbio_datasets()
    nb_datasets = len(names)
    print(f"Found {nb_datasets} datasets: {names}")
