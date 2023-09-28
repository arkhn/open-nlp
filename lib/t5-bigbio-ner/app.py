import re
from typing import List

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()

# Load the T5 model and tokenizer from the Hydra config
tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained("rntc/t5-instructionner-bigbio")


class Sentence(BaseModel):
    text: str


class Entities(BaseModel):
    entities: List[str]


def format_entity(entity: str) -> str:
    return f'- "{entity}"'


def is_valid_format(text: str) -> bool:
    pattern = r'^- ".*"$'
    return bool(re.match(pattern, text))


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/t5", response_class=PlainTextResponse)
async def extract_entities(sentence: Sentence):
    """
    Extract entities from a sentence using a T5 model fine-tuned on the BigBio NER dataset.

    Args:
        sentence (Sentence): Input sentence to extract entities from.

        Returns:
            PlainTextResponse: Response with extracted entities.
    """
    prompt = f"""Sentence: {sentence.text}
    Instructions: please extract entity words from the input sentence
    """

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=512)
    output_entities = tokenizer.decode(output[0], skip_special_tokens=True).split(", ")

    formatted_entities = [format_entity(entity) for entity in output_entities]
    valid_entities = [text for text in formatted_entities if is_valid_format(text)]

    response_text = "\n".join(valid_entities)
    return PlainTextResponse(content=response_text, status_code=200)
