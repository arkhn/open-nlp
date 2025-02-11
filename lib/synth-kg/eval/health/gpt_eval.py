import argparse
import asyncio
import copy
import json
import logging
import os
import random
import re
import time

import openai
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

random.seed(42)


def load_or_convert_to_dataframe(dataset_path):
    if "jsonl" in dataset_path:
        dataset = [json.loads(line) for line in open(dataset_path, "r")]
        # import pdb;pdb.set_trace()
    elif "json" in dataset_path:
        with open(dataset_path, "r") as file:
            dataset = json.load(file)
    elif "parquet" in dataset_path:
        dataset = pd.read_parquet(dataset_path)
        dataset = dataset.to_dict("records")
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
    return dataset


async def eval_dispatch_openai_requests(
    messages_list,
    model,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    timeout_seconds=10,
    base_wait_time=5,  # Base wait time in seconds
    backoff_factor=1.5,  # # Adding a new parameter for timeout
):
    """
    Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        frequency_penalty: Frequency penalty to use for the model.
        presence_penalty: Presence penalty to use for the model.
        timeout_seconds: Maximum number of seconds to wait for a response.

    Returns:
        List of responses from OpenAI API.
    """

    async def send_request(message):
        return await aclient.chat.completions.create(
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    async def request_until_success(message):
        while True:
            wait_time = base_wait_time
            try:
                return await asyncio.wait_for(send_request(message), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                print(f"Timeout! Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)  # Wait for the calculated time
                wait_time *= backoff_factor  # Increase the wait time
            except openai.BadRequestError as e:
                print(f"Bad request error: {str(e)}, message: {message}")
                return None

    async_responses = [request_until_success(x) for x in messages_list]
    return await asyncio.gather(*async_responses)


def eval_encode_prompt(prompt, instruction, model_output, reference_output, args):
    """Encode multiple prompt instructions into a single string."""

    if args.reference_first:
        output_list = [reference_output, model_output]
    else:
        output_list = [model_output, reference_output]

    mapping_dict_output = {"instruction": instruction}
    mapping_dict_generator = {}
    for idx in range(2):
        mapping_dict_output["output_" + str(idx + 1)] = output_list[idx]["output"]
        mapping_dict_generator["model_" + str(idx + 1)] = output_list[idx]["generator"]

    filled_prompt = eval_make_prompt(prompt, mapping_dict_output)

    return filled_prompt, mapping_dict_generator


def eval_make_prompt(template, val_dict):
    # flake8: noqa: W605
    text_to_format = re.findall("{([^ \s]+?)}", template)
    prompt = copy.deepcopy(template)
    for to_format in text_to_format:
        prompt = prompt.replace("{" + to_format + "}", val_dict[to_format], 1)

    return prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/comparsion/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        default=None,
        help="The path to the model output data.",
    )
    parser.add_argument(
        "--reference_output",
        type=str,
        required=True,
        default="./dataset/alpaca_eval/alpaca_eval.json",
        help="The path to the reference output data.",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        required=False,
        default=None,
        help="The file name for output.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        default=None,
        help="The model comparsion name.",
    )
    parser.add_argument(
        "--refer_model_name",
        type=str,
        required=False,
        default="text-davinci-003",
        help="The reference model name.",
    )
    parser.add_argument("--engine", type=str, default="davinci", help="The engine to use.")
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=20,
        help="The number of requests to send to GPT3 at a time.",
    )
    parser.add_argument("--max_tokens", type=int, default=100, help="Max input tokens.")
    parser.add_argument("--retries", type=int, default=5, help="failed retry times.")
    parser.add_argument(
        "--reference_first",
        action="store_true",
        help="If pass reference model will be model_1, otherwise reference model will be model_2.",
    )

    parser.add_argument(
        "--task_name", type=str, default=None, help="The task_name for different instruction usage."
    )

    parser.add_argument(
        "--max_test_number",
        type=int,
        default=-1,
        help="set if the test instances is less than generation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # import pdb;pdb.set_trace()
    model_output = load_or_convert_to_dataframe(args.model_output)
    if args.max_test_number != -1:
        model_output = model_output[: args.max_test_number]
    reference_output = load_or_convert_to_dataframe(args.reference_output)[: args.max_test_number]

    if args.task_name == "alpaca_eval" or args.task_name == "medsi":
        instructions = [item["instruction"] for item in reference_output]
    elif args.task_name == "iCliniq":
        instructions = [item["instruction"] for item in model_output]
    else:
        raise ValueError("Unsupported task.")

    if "response" in model_output[0]:
        model_output = [
            {"generator": args.model_name, "output": item["response"]} for item in model_output
        ]
    else:
        model_output = [
            {"generator": args.model_name, "output": item["output"]} for item in model_output
        ]
    if "response" in reference_output[0]:
        reference_output = [
            {"generator": args.refer_model_name, "output": item["response"]}
            for item in reference_output
        ]
    else:
        reference_output = [
            {"generator": args.refer_model_name, "output": item["output"]}
            for item in reference_output
        ]
    # import pdb;pdb.set_trace()
    assert len(model_output) == len(reference_output) == len(instructions)
    total = len(reference_output)
    progress_bar = tqdm(total=total)

    wait_base: float = 10
    retry_cnt = 0
    batch_size = args.request_batch_size

    if args.engine == "gpt-3.5-turbo":
        system_prompt = (
            "You are a helpful instruction-following assistant that prints "
            "the best model by selecting the best outputs for a given instruction."
        )
        prompt = open("eval/health/alpaca_eval_chat_gpt.txt").read() + "\n"
    else:
        raise ValueError("Unsupported engine.")

    results: list[dict] = []
    target_length = args.max_tokens

    if not os.path.exists(args.batch_dir):
        os.makedirs(args.batch_dir)

    if args.output_file_name is None:
        if args.reference_first:
            args.output_file_name = (
                args.model_name
                + "_"
                + args.refer_model_name
                + "_"
                + args.engine
                + "_"
                + str(args.max_test_number)
                + "_reference_first.jsonl"
            )
        else:
            args.output_file_name = (
                args.model_name
                + "_"
                + args.refer_model_name
                + "_"
                + args.engine
                + "_"
                + str(args.max_test_number)
                + "_reference_last.jsonl"
            )
    output_path = os.path.join(args.batch_dir, args.output_file_name)
    outputs = []
    if os.path.isfile(output_path):
        with open(output_path, "r") as fin:
            for line in fin:
                prev_output = json.loads(line)
                outputs.append(prev_output)
        print(f"Loaded {len(outputs)} machine-generated instructions")

    print(output_path)
    progress_bar.update(len(outputs))

    idx = len(outputs)
    with open(output_path, "a") as fout:
        while idx < total:
            message_list = []
            model2name_list = []
            j = 0
            while j < min(batch_size, total - idx):
                instr, m_o, r_o = (
                    instructions[idx + j],
                    model_output[idx + j],
                    reference_output[idx + j],
                )
                task_prompt, model2name = eval_encode_prompt(prompt, instr, m_o, r_o, args)
                message = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": task_prompt,
                    },
                ]
                message_list.append(message)
                model2name_list.append(model2name)
                j += 1
            batch_results: list[dict] = []
            while len(batch_results) == 0:
                try:
                    batch_predictions = asyncio.run(
                        eval_dispatch_openai_requests(
                            messages_list=message_list,
                            model=args.engine,
                            temperature=0,
                            max_tokens=target_length,
                            top_p=1.0,
                            frequency_penalty=0,
                            presence_penalty=0,
                        )
                    )
                    for j, message in enumerate(message_list):
                        data = {
                            "prompt": message[1]["content"],
                            "response": (
                                batch_predictions[j].choices[0].message.content
                                if batch_predictions[j]
                                else "a"
                            ),
                        }
                        batch_results.append(data)

                    # predictions += batch_results

                    retry_cnt = 0
                    break

                except openai.OpenAIError as e:
                    print(f"OpenAIError: {e}.")
                    if "Please reduce the length of the messages or completion" in str(e):
                        target_length = int(target_length * 0.8)
                        print(f"Reducing target length to {target_length}, retrying...")
                    else:
                        retry_cnt += 1
                        print("retry number: ", retry_cnt)
                        time.sleep(wait_base)
                        wait_base = wait_base * 1.5

            for model2name, result in zip(model2name_list, batch_results):
                merged_dict = {**model2name, **result}
                fout.write(json.dumps(merged_dict) + "\n")
                outputs.append(merged_dict)
                progress_bar.update(1)
                idx += 1
