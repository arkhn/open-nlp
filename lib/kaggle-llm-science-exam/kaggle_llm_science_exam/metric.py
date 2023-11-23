import string


def score_batch(
    generated_answers_lists: list[list[str]],
    reference_answers: list[str],
) -> list[float]:
    """Score a batch of generated answers against a batch of reference answers.

    Args:
        generated_answers_lists: Lists of generated answers (several generated answers correspond
            to each question).
        reference_answers: Reference answers.

    Returns:
        List of scores (one score per question).
    """
    if not len(generated_answers_lists) == len(reference_answers):
        raise ValueError("Expected the same number of generated answers and reference answers")

    batch_scores: list[float] = []
    for generated_answers, reference_answer in zip(generated_answers_lists, reference_answers):
        # remove duplicates and invalid answers in generated answers
        new_generated_answers = []
        for answer in generated_answers:
            # preprocess the results
            answer = answer.strip()
            answer = answer.translate(str.maketrans("", "", string.punctuation))
            answer = answer.upper()

            if answer in new_generated_answers:  # duplicates
                continue
            if answer not in {"A", "B", "C", "D", "E"}:  # invalid answer
                continue
            new_generated_answers.append(answer)

        # truncate & pad answers to have exactly 3 answers
        new_generated_answers = new_generated_answers[:3]
        while len(new_generated_answers) < 3:
            new_generated_answers.append("NA")

        # compute the average precision at 3
        score: float = 0
        for i, answer in enumerate(new_generated_answers):  # k is i+1
            if answer != reference_answer:
                continue
            precision_at_k = 1 / (i + 1)  # shortcut assuming that there is only one good answer
            score += precision_at_k
        batch_scores.append(score)

    return batch_scores
