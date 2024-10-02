import hydra
import pandas as pd
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from omegaconf import DictConfig, OmegaConf
from pandas import json_normalize
from tqdm import tqdm

load_dotenv()


def is_yes(result):
    return "yes" in result[:5].lower()


def generate_data(row, transcription_chain, question_chain):
    synthetic_transcription = transcription_chain.invoke(
        {"instruction": row["instruction"], "description": row["description"]}
    ).strip()

    data = []
    for _ in range(2):
        real_question = question_chain.invoke({"transcription": row["transcription"]}).strip()
        synthetic_question = question_chain.invoke(
            {"transcription": synthetic_transcription}
        ).strip()

        data.append(
            {
                "question": real_question,
                "synthetic_question": synthetic_question,
                "answer": "yes",
                "synthetic_answer": "yes",
                "synthetic_transcription": synthetic_transcription,
                "transcription": row["transcription"],
            }
        )
    return data


def evaluate(row, evaluation_chain):
    results = {}
    for i in ["synthetic_transcription", "transcription"]:
        for j in ["synthetic_question", "question"]:
            results[f"{i}/{j}"] = evaluation_chain.invoke(
                {"transcription": row[i], "question": row[j]}
            )

    raw_score = sum(1 for result in results.values() if is_yes(result))
    synthetic_score = 1 if is_yes(results["synthetic_transcription/synthetic_question"]) else 0
    real_score = 1 if is_yes(results["transcription/question"]) else 0
    strict_synthetic_qa_score = (
        1
        if is_yes(results["synthetic_transcription/synthetic_question"])
        and is_yes(results["transcription/synthetic_question"])
        else 0
    )
    strict_qa_score = (
        1
        if is_yes(results["transcription/question"])
        and is_yes(results["synthetic_transcription/question"])
        else 0
    )
    results["synthetic_score"] = synthetic_score
    results["raw_score"] = raw_score / 4
    results["real_score"] = real_score
    results["strict_synthetic_qa_score"] = strict_synthetic_qa_score
    results["strict_qa_score"] = strict_qa_score
    return results


def create_chain(template, input, llm):
    prompt = PromptTemplate(input=input, template=template)
    return prompt | llm | StrOutputParser()


@hydra.main(config_path="./configs", config_name="run.yaml")
def main(cfg: DictConfig):
    wandb.init(project="document-cross-validation", entity="clinical-dream-team")
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    llm = hydra.utils.instantiate(cfg.model)
    transcription_chain = create_chain(
        cfg.prompts.transcription,
        ["instruction", "description"],
        llm,
    )
    question_chain = create_chain(
        cfg.prompts.question,
        ["transcription"],
        llm,
    )
    evaluation_chain = create_chain(
        cfg.prompts.evaluation,
        ["transcription", "question"],
        llm,
    )

    # Load and process dataset
    loaded_dataset = load_dataset("DataFog/medical-transcription-instruct", split="train")
    df = loaded_dataset.to_pandas().iloc[: cfg.samples]

    tqdm.pandas(desc="Generating data...")

    ds_questions = [
        item
        for _, row in df.progress_apply(
            generate_data,
            axis=1,
            args=[transcription_chain, question_chain],
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
            df_joined[f"{score_type}_score"].agg(stat)
            if stat == "sum"
            else df_joined[f"{score_type}_score"].agg(stat) / len(df_joined)
        )
        for stat in ["sum", "mean"]
        for score_type in ["raw", "synthetic", "real", "strict_synthetic_qa", "strict_qa"]
    }
    for key, value in log_dict.items():
        wandb.run.summary[key] = value
    wandb.log({"dataset/evaluation": wandb.Table(dataframe=df_joined)})

    wandb.finish()


if __name__ == "__main__":
    main()
