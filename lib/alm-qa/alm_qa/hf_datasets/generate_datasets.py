import json.decoder

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
max_length = 1024
number_of_generations = 5
max_texts = 3


def main():
    # load datasets
    mimic_iii = datasets.load_dataset("bio-datasets/mimic_style_transfer")

    # preprocess mimic_iii dataset
    def hpi(example):
        example["text"] = example["text"][0]

    mimic_iii = mimic_iii["train"].map(hpi)
    mimic_iii = mimic_iii.select(list[range(max_texts)])

    # load the model mistral-7b
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        use_flash_attention_2=True,
        device_map="auto",
    )
    tokenizers = AutoTokenizer.from_pretrained(model_name)

    # create the prompt
    first_prompt = """
    You are a medical expert and given this clinical report, 
    you have to generate a pertinent question and response related 
    to this patient's history of present illness section :

    {}

    """

    # generate the questions and answers
    def generate_qa(example):
        example["questions"] = []
        example["answers"] = []
        messages = [
            {"role": "user", "content": first_prompt.format(data_point["text"])},
        ]

        # generate certain number of questions and answers
        for i in range(number_of_generations):
            model_inputs = tokenizers.apply_chat_template(messages, return_tensors="pt")
            generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
            decoded = tokenizers.batch_decode(generated_ids)
            try:

                # decode the generated ids if json format is well-formed.
                qa = json.loads(decoded[0])

            except Exception:
                print(decoded[0])
                continue

            example["questions"].append(qa["Q"])
            example["answers"].append(qa["A"])

    mimic_iii.map(generate_qa)
    mimic_iii.to_json("mimic_qa.json")
    # mimic_iii.to_parquet("mimic_qa.parquet")


if __name__ == "__main__":
    main()
