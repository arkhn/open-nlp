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


def create_synthetic_summary(summary_chain, row):
    """
    Create synthetic summary

    Args:
        summary_chain: The chain for generating summaries
        row: The row to generate the summary for

    Returns:
        The synthetic summary
    """
    return summary_chain.invoke(
        {
            "section_title": row["section_title"],
            "text": row["text"],
        }
    ).strip()


def create_question(question_chain, column, num_questions):
    """
    Create real/synthetic questions

    Args:
        question_chain: The chain for generating questions
        row: The row to generate the question for (optiont: real or synthetic data)

    Returns:
        The generated questions
    """
    return question_chain.invoke(
        {
            "summary": column,
            "num_questions": num_questions,
        }
    )


def generate_data(
    row: pd.Series,
    num_questions: int,
    summary_chain,
    question_chain,
):
    """
    Generate data for a given row in the dataset

    Args:
        row: text data
        num_questions: number of questions to generate
        summary_chain: generated synthetic summary
        question_chain: generated synthetic questions

    Returns:
        The merged data with the following columns:
        question, synthetic_question, answer, synthetic answer,
        synthetic_summary, summary

    """
    # Generate synthetic summary
    synthetic_summary = create_synthetic_summary(summary_chain, row)

    # Generate questions for the real and synthetic summaries
    real_question = create_question(question_chain, row["summary"], num_questions)
    synthetic_question = create_question(question_chain, synthetic_summary, num_questions)

    min_length = min(len(real_question), len(synthetic_question))
    real_question = real_question[:min_length]
    synthetic_question = synthetic_question[:min_length]

    data = []
    for sq, q in zip(synthetic_question, real_question):
        data.append(
            {
                "question": q["question"],
                "synthetic_question": sq["question"],
                "answer": "yes",
                "synthetic_answer": "yes",
                "synthetic_summary": synthetic_summary,
                "summary": row["summary"],
            }
        )

    return data


def compute_conformity(
    synthetic_summary_question: str,
    summary_synthetic_question: str,
    summary_question: str,
    synthetic_summary_synthetic_question: str,
) -> float:
    """
    Calculate conformity score.
    It is derived by identifying the percentage of questions
    for which the summary’s answer is "NO" and the document’s
    is "YES", or vice versa, and computing 100 − X

    Args:
        synthetic_summary_question: Synthetic summary for groundtruth question
        summary_synthetic_question: Groundtruth summary for synthetic question
        summary_question: Groundtruth summary for groundtruth question
        synthetic_summary_synthetic_question: Synthetic summary for synthetic question

    Returns:
        The conformity score
    """
    score = 2
    if (
        is_yes(synthetic_summary_question) != is_yes(summary_question)
        or is_idk(synthetic_summary_question) != is_idk(summary_question)
        or is_no(synthetic_summary_question) != is_no(summary_question)
    ):
        score -= 1
    if (
        is_yes(summary_synthetic_question) != is_yes(synthetic_summary_synthetic_question)
        or is_idk(summary_synthetic_question) != is_idk(synthetic_summary_synthetic_question)
        or is_no(summary_synthetic_question) != is_no(synthetic_summary_synthetic_question)
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
        ["synthetic_summary", "summary"],
        ["synthetic_question", "question"],
    ):
        results[f"{i}/{j}"] = evaluation_chain.invoke({"summary": row[i], "question": row[j]})

    # Compute conformity, consistency and coverage
    coverage = 1 if is_idk(results["synthetic_summary/question"]) else 0
    consistency = 1 if not is_idk(results["summary/synthetic_question"]) else 0
    conformity = compute_conformity(
        results["synthetic_summary/question"],
        results["summary/synthetic_question"],
        results["summary/question"],
        results["synthetic_summary/synthetic_question"],
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
        question generation (True) or summary generation (False)

    Returns:
        The chain of models for either question or summary generation
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


def merge_evaluate_df(df: pd.DataFrame, row: pd.Series, evaluation_chain) -> pd.DataFrame:
    """
    Evaluate the generated data

    Args:
        df: The dataframe to evaluate
        row: The row to evaluate
        evaluation_chain: The evaluation chain

    Returns:
        df: The concatenate dataframw with additional evaluation results,
        including conformity, consistency and coverage
    """
    # Evaluate
    tqdm.pandas(desc="Evaluating...")
    row["evaluation"] = row.progress_apply(
        evaluate,
        args=[evaluation_chain],
        axis=1,
    )
    json_df = json_normalize(row["evaluation"])

    # Combine the original dataframe with the extracted JSON data
    row = pd.concat([row, json_df], axis=1)
    del row["evaluation"]

    # Join df_questions and df
    df_joined = df.merge(row, left_on="summary", right_on="summary", how="right")
    print(f"Shape of joined dataframe: {df_joined.shape}")
    return df_joined


def log_wandb(df: pd.DataFrame) -> dict:
    """
    Log the results in wandb
    Args:
        df: The dataframe to log
    Returns:
        log_dict: The dictionary containing the results
    """
    log_dict = {
        f"{stat}/score/{score_type}": (
            df[f"{score_type}"].agg(stat)
            if stat == "sum"
            else df[f"{score_type}"].agg("sum") / len(df)
        )
        for stat in ["sum", "mean"]
        for score_type in ["consistency", "conformity", "coverage"]
    }
    return log_dict


def process_questions(df: pd.DataFrame, num_questions: int, summary_chain, question_chain):
    """
    Generate questions for the the summary

    Args:
        df: The dataframe to generate questions for
        num_questions: The number of questions to generate
        summary_chain: The chain for generating summaries
        question_chain: The chain for generating questions

    Returns:
        ds_questions: The generated questions
    """
    ds_questions = [
        item
        for _, row in df.progress_apply(
            generate_data,
            axis=1,
            args=[num_questions, summary_chain, question_chain],
        ).items()
        for item in row
    ]
    return ds_questions


@hydra.main(config_path="./configs", config_name="run_mimoracle.yaml")
def main(cfg: DictConfig):
    # Initialize WandB and log the models
    wandb.init(project="document-cross-validation", entity="clinical-dream-team")
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    llm = hydra.utils.instantiate(cfg.model)
    question_llm = hydra.utils.instantiate(cfg.question_model)
    summary_chain = create_chain(cfg.prompts.summary, llm, False)
    question_chain = create_chain(cfg.prompts.question, question_llm, True)
    evaluation_chain = create_chain(cfg.prompts.evaluation, question_llm, False)

    # Load and process dataset
    loaded_dataset = load_dataset(cfg.dataset, split="train")
    df = loaded_dataset.to_pandas().iloc[: cfg.samples]

    tqdm.pandas(desc="Generating data...")
    ds_questions = process_questions(df, cfg.num_questions, summary_chain, question_chain)
    print(f"Shape of generated data: {len(ds_questions)}")
    df_questions = pd.DataFrame(ds_questions)

    # Evaluate
    df_joined = merge_evaluate_df(df, df_questions, evaluation_chain)

    # Log results in wandb
    log_dict = log_wandb(df_joined)
    for key, value in log_dict.items():
        wandb.run.summary[key] = value
    wandb.log({"dataset/evaluation_mimoracle_gpt4o_retest": wandb.Table(dataframe=df_joined)})
    wandb.finish()


if __name__ == "__main__":
    main()
