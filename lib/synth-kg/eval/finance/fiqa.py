import warnings
from pathlib import Path

import datasets
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from vllm import SamplingParams

warnings.filterwarnings("ignore")
with open(Path(__file__).parent / "sentiment_templates.txt") as f:
    templates = [line.strip() for line in f.readlines()]


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def add_instructions(x):
    if x.format == "post":
        return (
            "What is the sentiment of this tweet? Please choose an answer "
            "from {negative/neutral/positive}."
        )
    else:
        return (
            "What is the sentiment of this news? Please choose an answer "
            "from {negative/neutral/positive}."
        )


def make_label(x):
    if x < -0.1:
        return "negative"
    elif x >= -0.1 and x < 0.1:
        return "neutral"
    elif x >= 0.1:
        return "positive"


def change_target(x):
    if "positive" in x or "Positive" in x:
        return "positive"
    elif "negative" in x or "Negative" in x:
        return "negative"
    else:
        return "neutral"


def vote_output(x):
    output_dict = {"positive": 0, "negative": 0, "neutral": 0}
    for i in range(len(templates)):
        pred = change_target(x[f"out_text_{i}"].lower())
        output_dict[pred] += 1
    if output_dict["positive"] > output_dict["negative"]:
        return "positive"
    elif output_dict["negative"] > output_dict["positive"]:
        return "negative"
    else:
        return "neutral"


def test_fiqa(args, model, prompt_fun=add_instructions):
    batch_size = args.batch_size
    # dataset = load_dataset('pauri32/fiqa-2018')
    dataset = load_from_disk("eval/finance/data/fiqa-2018/")
    dataset = datasets.concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )
    dataset = dataset.train_test_split(0.226, seed=42)["test"]
    dataset = dataset.to_pandas()
    dataset["output"] = dataset.sentiment_score.apply(make_label)
    if prompt_fun is None:
        dataset["instruction"] = (
            "What is the sentiment of this news? Please choose an answer from "
            "{negative/neutral/positive}."
        )
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis=1)

    dataset = dataset[["sentence", "output", "instruction"]]
    dataset.columns = ["input", "output", "instruction"]
    dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset["context"].tolist()
    total_steps = dataset.shape[0] // batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    for i in tqdm(range(total_steps)):
        tmp_context = context[i * batch_size : (i + 1) * batch_size]
        outputs = model.generate(tmp_context, sampling_params, use_tqdm=False)
        res_sentences = [output.outputs[0].text for output in outputs]
        tqdm.write(f"{i}: {res_sentences[0]}")
        out_text = [o for o in res_sentences]
        out_text_list += out_text

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])
    f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average="macro")
    f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average="micro")
    f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average="weighted")

    print(
        f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted "
        f"(BloombergGPT): {f1_weighted}. "
    )

    return dataset


def test_fiqa_mlt(args, model):
    batch_size = args.batch_size
    # dataset = load_dataset('pauri32/fiqa-2018')
    dataset = load_from_disk("eval/finance/data/fiqa-2018/")
    dataset = datasets.concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )
    dataset = dataset.train_test_split(0.226, seed=42)["test"]
    dataset = dataset.to_pandas()
    dataset["output"] = dataset.sentiment_score.apply(make_label)
    dataset["text_type"] = dataset.apply(
        lambda x: "tweet" if x.format == "post" else "news", axis=1
    )
    dataset = dataset[["sentence", "output", "text_type"]]
    dataset.columns = ["input", "output", "text_type"]

    dataset["output"] = dataset["output"].apply(change_target)
    dataset = dataset[dataset["output"] != "neutral"]

    out_texts_list = [[] for _ in range(len(templates))]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_length)

    for i, template in enumerate(templates):
        dataset = dataset[["input", "output", "text_type"]]
        dataset["instruction"] = dataset["text_type"].apply(
            lambda x: template.format(type=x) + "\nOptions: positive, negative"
        )
        dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")

        contexts = dataset["context"].tolist()
        total_steps = len(contexts) // batch_size + 1

        for idx in tqdm(range(total_steps)):
            batch_contexts = contexts[idx * batch_size : (idx + 1) * batch_size]
            outputs = model.generate(batch_contexts, sampling_params)
            res_sentences = [output.outputs[0].text for output in outputs]
            tqdm.write(f"{idx}: {res_sentences[0]}")
            out_text = [o.split("Answer: ")[1] for o in res_sentences]
            out_texts_list[i] += out_text

    for i in range(len(templates)):
        dataset[f"out_text_{i}"] = out_texts_list[i]
        dataset[f"out_text_{i}"] = dataset[f"out_text_{i}"].apply(change_target)

    dataset["new_out"] = dataset.apply(vote_output, axis=1, result_type="expand")

    dataset.to_csv("tmp.csv")

    for k in [f"out_text_{i}" for i in range(len(templates))] + ["new_out"]:
        acc = accuracy_score(dataset["target"], dataset[k])
        f1_macro = f1_score(dataset["target"], dataset[k], average="macro")
        f1_micro = f1_score(dataset["target"], dataset[k], average="micro")
        f1_weighted = f1_score(dataset["target"], dataset[k], average="weighted")

        print(
            f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted "
            f"(BloombergGPT): {f1_weighted}. "
        )

    return dataset
