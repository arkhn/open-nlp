import pytest
from kaggle_llm_science_exam.metric import score_batch


@pytest.mark.parametrize(
    "generated_answers_lists, reference_answers, batch_scores",
    [
        (
            [["A", "B"], ["A", "B", "C"], ["A", "B", "C", "D"]],
            ["A", "B", "C"],
            [1, 1 / 2, 1 / 3],
        ),
        (
            [["A", "B"], ["a", "b", "c"], [" (a) ", " (a) ", " (c) ", " (d) "]],
            ["C", "B", "D"],
            [0, 1 / 2, 1 / 3],
        ),
    ],
)
def test_score_batch(
    generated_answers_lists: list[list[str]],
    reference_answers: list[str],
    batch_scores: list[float],
) -> None:
    output_batch_scores: list[float] = score_batch(
        generated_answers_lists=generated_answers_lists,
        reference_answers=reference_answers,
    )
    assert output_batch_scores == batch_scores
