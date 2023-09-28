PROMPT = (
    "Instruction: extract the exact match of disorders, diseases or symptoms mentioned"
    " in the text or return None if there is no clinical entity:\n"
)


def promptify(dataset, tags_dict):
    """Create few-shot examples in a prompt format.

    Args:
        dataset: Dataset to use.
        tags_dict: Dictionary of tags.

    Yields:
        Prompt in a string format.
    """

    none_statement = '- "None"\n'
    for example in dataset:
        chunked_tokens_examples = []
        current_tag = None
        current_chunk = ""
        text = ""
        tags = ""
        # Iterate over the tokens and tags of the example
        for tag, token in zip(example["clinical_entity_tags"], example["tokens"]):
            str_tag = tags_dict(tag)
            # Merge the tokens in IOB format
            if str_tag.startswith("B-"):
                if current_tag:
                    chunked_tokens_examples.append(current_chunk[:-1])
                current_chunk = token + " "
                current_tag = str_tag[2:]
            elif str_tag.startswith("I-"):
                current_chunk += token + " "
        if current_tag:
            chunked_tokens_examples.append(current_chunk[:-1])
        # Convert tag in a bullet list string format
        tags += "".join([f'- "{token}"\n' for token in chunked_tokens_examples])
        text += " " + example["text"]
        # Print the prompt when the number of examples to concat is reached
        prompt_example = f"Input: {text}\n" f"{PROMPT}" f"{tags if tags else none_statement}\n"
        yield prompt_example
