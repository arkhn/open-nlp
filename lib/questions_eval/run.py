import itertools

import hydra
import pandas as pd
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser
from langchain.schema import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from omegaconf import DictConfig, OmegaConf
from pandas import json_normalize
from tqdm import tqdm

load_dotenv()


def is_yes(result: str) -> bool:
    """
    Check if the result is a yes

    Args:
        result: The result to check

    Returns:
        True if the result is a yes, False otherwise
    """
    return True if "yes" in result[:5].lower() else False


def is_idk(result: str) -> bool:
    """
    Check if the result is a 'idk' (i don't know)

    Args:
        result: The result to check

    Returns:
        True if the result is a idk, False otherwise
    """
    return True if "idk" in result[:5].lower() else False


def is_no(result: str) -> bool:
    """
    Check if the result is a no

    Args:
        result: The result to check

    Returns:
        True if the result is a no, False otherwise
    """
    return True if "no" in result[:5].lower() else False


def generate_data(
    row: pd.Series,
    num_questions: int,
    transcription_chain,
    question_chain,
):
    """
    Generate data for a given row in the dataset

    Args:
        row: text data
        num_questions: number of questions to generate
        transcription_chain: generated synthetic transcription
        question_chain: generated synthetic questions

    Returns:
        The merged data with the following columns:
        question, synthetic_question, answer, synthetic answer,
        synthetic_transcription, transcription

    """
    synthetic_transcription = transcription_chain.invoke(
        {
            "keywords": row["keywords"],
            "derived_keywords": row["derived_keywords"],
            "description": row["description"],
            "medical_specialty": row["medical_specialty"],
        }
    ).strip()

    data = []
    real_question = question_chain.invoke(
        {
            "transcription": row["transcription"],
            "num_questions": num_questions,
        }
    )
    synthetic_question = question_chain.invoke(
        {
            "transcription": synthetic_transcription,
            "num_questions": num_questions,
        }
    )

    min_length = min(len(real_question), len(synthetic_question))
    real_question = real_question[:min_length]
    synthetic_question = synthetic_question[:min_length]

    for sq, q in zip(synthetic_question, real_question):
        data.append(
            {
                "question": q["question"],
                "synthetic_question": sq["question"],
                "answer": "yes",
                "synthetic_answer": "yes",
                "synthetic_transcription": synthetic_transcription,
                "transcription": row["transcription"],
            }
        )

    return data


def compute_conformity(
    synthetic_transcript_question: str,
    transcript_synthetic_question: str,
    transcript_question: str,
    synthetic_transcript_synthetic_question: str,
) -> float:
    """
    Calculate conformity score.
    It is derived by identifying the percentage of questions
    for which the summary’s answer is "NO" and the document’s
    is "YES", or vice versa, and computing 100 − X

    Args:
        synthetic_transcript_question: Synthetic transcript for groundtruth question
        transcript_synthetic_question: Groundtruth transcript for synthetic question
        transcript_question: Groundtruth transcript for groundtruth question
        synthetic_transcript_synthetic_question: Synthetic transcript for synthetic question

    Returns:
        The conformity score
    """
    score = 2
    if (
        is_yes(synthetic_transcript_question) != is_yes(transcript_question)
        or is_idk(synthetic_transcript_question) != is_idk(transcript_question)
        or is_no(synthetic_transcript_question) != is_no(transcript_question)
    ):
        score -= 1
    if (
        is_yes(transcript_synthetic_question) != is_yes(synthetic_transcript_synthetic_question)
        or is_idk(transcript_synthetic_question) != is_idk(synthetic_transcript_synthetic_question)
        or is_no(transcript_synthetic_question) != is_no(synthetic_transcript_synthetic_question)
    ):
        score -= 1
    return float(score) / 2


def evaluate(row, evaluation_chain) -> dict:
    """
    Evaluate the generated data

    Args:
        row: The row to evaluate
        evaluation_chain: The evaluation chain

    Returns:
        The evaluation results, including conformity, consistency and coverage
    """
    results = {}
    for i, j in itertools.product(
        ["synthetic_transcription", "transcription"],
        ["synthetic_question", "question"],
    ):
        results[f"{i}/{j}"] = evaluation_chain.invoke({"transcription": row[i], "question": row[j]})

    # Compute conformity, consistency and coverage
    coverage = 1 if is_idk(results["synthetic_transcription/question"]) else 0
    consistency = 1 if not is_idk(results["transcription/synthetic_question"]) else 0
    conformity = compute_conformity(
        results["synthetic_transcription/question"],
        results["transcription/synthetic_question"],
        results["transcription/question"],
        results["synthetic_transcription/synthetic_question"],
    )
    results["consistency"] = consistency
    results["conformity"] = conformity
    results["coverage"] = coverage
    return results


def create_chain(template: str, llm: str, is_question_chain: bool):
    """
    Create a chain of models

    Args:
        template: The template for the prompt
        llm: The language model to use
        is_question_chain: Boolean indicating whether the chain is used for
        question generation (True) or transcription generation (False)

    Returns:
        The chain of models for either question or transcription generation
    """
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an helpful clinical assistant."),
            ("human", template),
        ]
    )
    return (
        chat_template
        | llm
        | (
            StrOutputParser()
            if not is_question_chain
            else OutputFixingParser.from_llm(llm, parser=JsonOutputParser())
        )
    )


@hydra.main(config_path="./configs", config_name="run.yaml")
def main(cfg: DictConfig):
    # Initialize WandB and log the models
    wandb.init(project="document-cross-validation", entity="clinical-dream-team")
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    llm = hydra.utils.instantiate(cfg.model)
    question_llm = hydra.utils.instantiate(cfg.question_model)
    transcription_chain = create_chain(cfg.prompts.transcription, llm, False)
    question_chain = create_chain(cfg.prompts.question, question_llm, True)
    evaluation_chain = create_chain(cfg.prompts.evaluation, question_llm, False)

    # Load and process dataset
    loaded_dataset = load_dataset(cfg.dataset, split="train")
    df = loaded_dataset.to_pandas().iloc[: cfg.samples]

    tqdm.pandas(desc="Generating data...")

    ds_questions = [
        item
        for _, row in df.progress_apply(
            generate_data,
            axis=1,
            args=[cfg.num_questions, transcription_chain, question_chain],
        ).items()
        for item in row
    ]

    df_questions = pd.DataFrame(ds_questions)

    # Evaluate
    tqdm.pandas(desc="Evaluating...")
    df_questions["evaluation"] = df_questions.progress_apply(
        evaluate,
        args=[evaluation_chain],
        axis=1,
    )
    json_df = json_normalize(df_questions["evaluation"])

    # Combine the original dataframe with the extracted JSON data
    df_questions = pd.concat([df_questions, json_df], axis=1)
    del df_questions["evaluation"]

    # Join df_questions and df
    df_joined = df.merge(
        df_questions, left_on="transcription", right_on="transcription", how="right"
    )
    print(f"Shape of joined dataframe: {df_joined.shape}")

    # Log results in wandb
    log_dict = {
        f"{stat}/score/{score_type}": (
            df_joined[f"{score_type}"].agg(stat)
            if stat == "sum"
            else df_joined[f"{score_type}"].agg("sum") / len(df_joined)
        )
        for stat in ["sum", "mean"]
        for score_type in ["consistency", "conformity", "coverage"]
    }
    for key, value in log_dict.items():
        wandb.run.summary[key] = value
    wandb.log({"dataset/evaluation": wandb.Table(dataframe=df_joined)})
    wandb.finish()


if __name__ == "__main__":
    main()
