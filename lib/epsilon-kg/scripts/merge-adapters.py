import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapters",
        type=str,
        help="Comma-separated list of adapters",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model to use",
        default="meta-llama/llama-2-7b-hf",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output merged model",
        default="./lora/merged",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    merge_adapters(args.model, args.adapters.split(","), args.output_path),


def merge_adapters(model_name, adapter_paths, output_path):
    model = AutoModelForCausalLM.from_pretrained(model_name)

    for adapter_path in adapter_paths:
        print(f"merge: {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    AutoTokenizer.from_pretrained(model_name).save_pretrained(output_path)
    model.save_pretrained(output_path)
    return output_path


if __name__ == "__main__":
    main()
