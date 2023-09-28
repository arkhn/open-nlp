from typing import List, Tuple


def generate_instruction(sentence: str, types: List[str], entities: List[str], task_type: str):
    """
    Generate instruction for a given sentence, entities, and types

    Args:
        sentence: Input sentence
        types: List of entity types
        entities: List of entity words
        task_type: Task type, one of "task_1", "task_2", "task_3"

    Returns:
        Dictionary with keys "prompt" and "answer"
    """

    def create_prompt() -> str:
        if task_type == "task_1":
            return f"""Sentence: {sentence}
            Instructions: please extract entities and their types from the input sentence, all entity types are in options
            Options: {options}
            """  # noqa
        elif task_type == "task_2":
            prompt_entities = ", ".join(entities)
            return f"""Sentence: {sentence}
            Instructions: please typing these entity words according to sentence: {prompt_entities}
            Options: {options}
            """
        elif task_type == "task_3":
            return f"""Sentence: {sentence}
            Instructions: please extract entity words from the input sentence
            """
        else:
            raise ValueError("Invalid task type")

    def create_answer() -> str:
        if task_type in ["task_1", "task_2"]:
            answer = ""
            for i in range(len(entities)):
                try:
                    if types[i][0] in "aeiou":
                        answer += f", {entities[i]} is an {types[i]}"
                    else:
                        answer += f", {entities[i]} is a {types[i]}"
                except ValueError:
                    print(f"Missing type for entity: {entities[i]}")
            return answer[2:] if answer.startswith(", ") else answer
        elif task_type == "task_3":
            return ", ".join(entities)
        else:
            raise ValueError("Invalid task type")

    if task_type in ["task_1", "task_2"]:
        unique_types = list(set(types))
        options = ", ".join(unique_types)
    else:
        options = None

    prompt = create_prompt()
    answer = create_answer()

    return {"prompt": prompt, "answer": answer}


def generate_merged_entities_and_types(
    tokens: List[str], ner_tags: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Generate merged entities and types from tokens and ner tags

    Args:
        tokens: List of tokens
        ner_tags: List of ner tags

    Returns:
        Tuple of lists of merged entities and types
    """
    merged_types = []
    merged_entities = []

    for i, ner_tag in enumerate(ner_tags):
        if ner_tag == "O":
            continue
        if ner_tag.startswith("B-"):
            merged_types.append(ner_tag[2:])
            merged_entities.append(tokens[i])
        else:
            merged_entities[-1] += " " + tokens[i]

    return merged_types, merged_entities


def generate_instruction_ner(row):
    """
    Generate instructions for NER task

    Args:
        row: row of a dataset with columns "tokenized_sentence", "ner_tags",
        "sentence", "entities_types", "id"

    Returns:
        Dictionary with key "instructions"
    """
    batch_instructions = []

    for i in range(len(row["tokenized_sentence"])):
        tokens = row["tokenized_sentence"][i]
        ner_tags = row["ner_tags"][i]
        sentence = row["sentence"][i]
        types = row["entities_types"][i]
        id = row["id"][i]

        merged_types, merged_entities = generate_merged_entities_and_types(tokens, ner_tags)

        instructions = [
            generate_instruction(sentence, merged_types, merged_entities, f"task_{i}")
            for i in range(1, 4)
        ]

        for i, instruction in enumerate(instructions):
            batch_instructions.append(
                {
                    "id": id + f"_task{i}",
                    "instruction": instruction["prompt"],
                    "answer": instruction["answer"],
                    "tokenized_sentence": tokens,
                    "ner_tags": ner_tags,
                    "sentence": sentence,
                    "entities_types": types,
                }
            )

    return {"instructions": batch_instructions}
