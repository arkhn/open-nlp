import glob
import pathlib
from typing import Optional

import pandas as pd
import six  # type: ignore
import typer
from google.cloud import translate_v3 as tr
from google.cloud.translate_v3.proto.translation_service_pb2 import TranslateTextGlossaryConfig
from tqdm import tqdm

app = typer.Typer()


def translate_text(
    text: str, project_id: Optional[str] = None, glossary_id: Optional[str] = None
) -> str:
    """Translates text into the target language.

    We use glossary to have no modification on the anonymised tokens
    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages

    Args:
        text: The text translate.
        project_id: The project-id inside GCP.
        glossary_id: The glossary-id inside GCP.
    """
    translate_client = tr.TranslationServiceClient()
    location = "us-central1"
    project_id = project_id
    glossary = translate_client.glossary_path(project_id, "us-central1", glossary_id)
    glossary_config = TranslateTextGlossaryConfig(glossary=glossary)
    parent = f"projects/{project_id}/locations/{location}"
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate_text(
        contents=[text],
        target_language_code="fr",
        source_language_code="en",
        parent=parent,
        mime_type="text/plain",
        glossary_config=glossary_config,
    )
    return result.glossary_translations[0].translated_text


@app.command()
def pricing(input_path: str):
    """Get the price of our corpus.

    Args:
        input_path : The corpus path folder containing all the text files.
    """
    char_count = 0
    for file in glob.glob(f"{input_path}/*.txt"):
        with open(file, "r") as txt:
            for line in txt.readlines():
                char_count += len(line)
    print(f"Google Translate API steals you {(20 * char_count) / 500_000}$ 💵")


@app.command()
def shuffle(input_path: str, output_path: str):
    """Shuffle the Mimic-III corpus. The granularity is at the document level
    Args:
        input_path: The Mimic-III corpus path in csv format.
        output_path: The destination of the shuffling.
    """
    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(input_path).sample(frac=1)
    df.to_csv(output_path)


@app.command()
def extract_mimic_texts(
    input_path: str,
    output_path: str,
    corpus_size_min: int,
    corpus_size_max: int,
):
    """Extract Mimic-III text from original csv file of Mimic-III.
    Args:
        input_path: The input path of the original csv file.
        output_path: The result containing the text files inside a folder.
        corpus_size_min: The index where to start the extraction.
        corpus_size_max: The index where to finish the extraction.
    """
    folder = f"{output_path}/{corpus_size_min}_{corpus_size_max}"
    pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(input_path)
    texts = df.TEXT[corpus_size_min:corpus_size_max]
    row_ids = df.ROW_ID[corpus_size_min:corpus_size_max]
    for text, row_id in tqdm(zip(texts, row_ids), total=len(texts)):
        with open(f"{folder}/{row_id}.txt", "w") as file:
            file.write(text)


@app.command()
def translate(
    input_path: str,
    output_path: str,
    glossary_id="mimic-iii-tokens",
    project_id="988404890916",
):
    """This function translate the Mimic-III corpus with GCP.
    This function wrapped the method to translate each extracting file extracted
    previously with GCP.
    Args:
        input_path: The folder containing all the extracted files.
        output_path: The output path containing the translation
        project_id: The GCP project id.
        glossary_id: The glossary id inside GCP project to keep the exact form of
            each anonymized token during translation.
    """
    folder = f"{output_path}/{pathlib.Path(input_path).name}"
    pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
    for file in tqdm(glob.glob(f"{input_path}/*.txt")):
        filename = pathlib.Path(file).name
        with open(file, "r") as file_to_translate:
            with open(f"{folder}/{filename}", "w") as translated_file:
                translated_file.write(
                    translate_text(
                        "".join(file_to_translate.readlines()),
                        glossary_id=glossary_id,
                        project_id=project_id,
                    )
                )


if __name__ == "__main__":
    app()
