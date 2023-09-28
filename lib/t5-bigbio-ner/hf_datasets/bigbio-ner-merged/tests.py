import spacy
from hydra import compose, initialize
from src.preprocess import preprocess_row

with initialize(config_path="conf", version_base="1.2"):
    config = compose(config_name="default")
nlp = spacy.load(config.tokenizer, disable=["tagger", "parser", "ner"])


def test_process_row_with_entities():
    """
    Test that the preprocess_row function correctly tokenizes a sentence and adds NER tags
    """
    sample_row = {
        "id": "1",
        "passages": [{"text": ["Aspirin is a drug."]}],
        "entities": [{"offsets": [[0, 7]], "type": "CLINICAL_ENTITY"}],
    }
    result = preprocess_row(sample_row, nlp)
    expected = {
        "id": "1",
        "tokenized_sentence": ["Aspirin", "is", "a", "drug", "."],
        "ner_tags": ["B-CLINICAL_ENTITY", "O", "O", "O", "O"],
    }
    result = {key: result[key] for key in expected.keys()}
    assert result == expected


def test_process_row_no_entities():
    """
    Test that the preprocess_row function correctly tokenizes a sentence and
    adds NER tags when there are no entities
    """
    sample_row = {
        "id": "2",
        "passages": [{"text": ["No entities here."]}],
        "entities": [],
    }
    result = preprocess_row(sample_row, nlp)
    expected = {
        "id": "2",
        "tokenized_sentence": ["No", "entities", "here", "."],
        "ner_tags": ["O", "O", "O", "O"],
    }
    result = {key: result[key] for key in expected.keys()}

    assert result == expected


def test_process_row_overlapping_entities():
    """
    Test that the preprocess_row function correctly tokenizes a sentence
    and adds NER tags when there are overlapping entities
    """
    sample_row = {
        "id": "3",
        "passages": [{"text": ["Aspirin and Aspirin-Advil are drugs."]}],
        "entities": [
            {"offsets": [[0, 7]], "type": "CLINICAL_ENTITY"},
            {"offsets": [[12, 19]], "type": "CLINICAL_ENTITY"},
            {"offsets": [[20, 25]], "type": "CLINICAL_ENTITY"},
            {"offsets": [[12, 25]], "type": "CLINICAL_ENTITY"},
        ],
    }
    result = preprocess_row(sample_row, nlp)
    expected = {
        "id": "3",
        "tokenized_sentence": [
            "Aspirin",
            "and",
            "Aspirin",
            "-",
            "Advil",
            "are",
            "drugs",
            ".",
        ],
        "ner_tags": [
            "B-CLINICAL_ENTITY",
            "O",
            "B-CLINICAL_ENTITY",
            "I-CLINICAL_ENTITY",
            "I-CLINICAL_ENTITY",
            "O",
            "O",
            "O",
        ],
    }
    result = {key: result[key] for key in expected.keys()}
    assert result == expected
