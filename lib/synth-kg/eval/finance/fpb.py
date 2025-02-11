import warnings
from pathlib import Path

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from vllm import SamplingParams

warnings.filterwarnings("ignore")
dic = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

with open(Path(__file__).parent / "sentiment_templates.txt") as f:
    templates = [line.strip() for line in f.readlines()]


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


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


def test_fpb(args, model, prompt_fun=None):
    batch_size = args.batch_size
    instructions = load_from_disk("eval/finance/data/financial_phrasebank-sentences_50agree")
    instructions = instructions["train"]
    instructions = instructions.train_test_split(seed=42)["test"]
    instructions = instructions.to_pandas()
    instructions.columns = ["input", "output"]
    instructions["output"] = instructions["output"].apply(lambda x: dic[x])

    if prompt_fun is None:
        instructions["instruction"] = (
            "What is the sentiment of this news? Please choose an answer from "
            "{negative/neutral/positive}."
        )
    else:
        instructions["instruction"] = instructions.apply(prompt_fun, axis=1)

    instructions[["context", "target"]] = instructions.apply(
        format_example, axis=1, result_type="expand"
    )

    print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")

    context = instructions["context"].tolist()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=28)

    out_text_list = []
    for i in tqdm(range(0, len(context), batch_size)):
        batch = context[i : i + batch_size]
        outputs = model.generate(batch, sampling_params, use_tqdm=False)
        for output in outputs:
            generated_text = output.outputs[0].text
            out_text_list.append(generated_text)

    instructions["out_text"] = out_text_list
    instructions["new_target"] = instructions["target"].apply(change_target)
    instructions["new_out"] = instructions["out_text"].apply(change_target)

    acc = accuracy_score(instructions["new_target"], instructions["new_out"])
    f1_macro = f1_score(instructions["new_target"], instructions["new_out"], average="macro")
    f1_micro = f1_score(instructions["new_target"], instructions["new_out"], average="micro")
    f1_weighted = f1_score(instructions["new_target"], instructions["new_out"], average="weighted")

    print(
        f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted "
        f"(BloombergGPT): {f1_weighted}. "
    )

    return instructions


def test_fpb_mlt(args, model):
    batch_size = args.batch_size
    dataset = load_from_disk("eval/finance/data/financial_phrasebank-sentences_50agree")
    dataset = dataset["train"]
    dataset = dataset.train_test_split(seed=42)["test"]
    dataset = dataset.to_pandas()
    dataset.columns = ["input", "output"]
    dataset["output"] = dataset["output"].apply(lambda x: dic[x])
    dataset["text_type"] = dataset.apply(lambda x: "news", axis=1)

    dataset["output"] = dataset["output"].apply(change_target)
    dataset = dataset[dataset["output"] != "neutral"]

    out_texts_list = [[] for _ in range(len(templates))]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    for i, template in enumerate(templates):
        dataset = dataset[["input", "output", "text_type"]]
        dataset["instruction"] = dataset["text_type"].apply(
            lambda x: template.format(type=x) + "\nOptions: positive, negative"
        )
        dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")

        contexts = dataset["context"].tolist()

        for idx in tqdm(range(0, len(contexts), batch_size)):
            batch = contexts[idx : idx + batch_size]
            outputs = model.generate(batch, sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text
                out_text = generated_text.split("Answer: ")[1]
                out_texts_list[i].append(out_text)

            if (idx + 1) % (len(contexts) // 5) == 0:
                tqdm.write(f"{idx}: {generated_text}")

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
