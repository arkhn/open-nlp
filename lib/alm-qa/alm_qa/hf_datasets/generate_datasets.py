import json.decoder

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
max_length = 1024
number_of_generations = 5
max_texts = 3
prompt = (
    "You are a medical expert that is given this clinical report, you have to generate one "
    "pertinent question related to the patient whose answer can be found in the report; "
    'the question and answer must be in this JSON format {"Q": "",  "A": ""}:\n'
)


def main():
    # load datasets
    mimic_iii = datasets.load_dataset("bio-datasets/mimic_style_transfer")
    ids = []
    texts = []
    for example in mimic_iii["train"]:
        for text_id, text in zip(example["text_id"], example["text"]):
            ids.append(f"{example['user_id']}-{text_id}")
            texts.append(text.replace("\n", " "))

    mimic_qa = datasets.Dataset.from_dict({"id": ids, "text": texts})
    mimic_qa = mimic_qa.select(range(max_texts))
    mimic_qa = mimic_qa.add_column("qa", [[]] * len(mimic_qa))

    # load the model mistral-7b
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        use_flash_attention_2=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # generate the questions and answers
    def generate_qa(example):
        messages = [
            {"role": "user", "content": prompt + example["text"][0]},
        ]

        # generate certain number of questions and answers
        for i in range(number_of_generations):
            model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
                model.device
            )
            generated_ids = model.generate(model_inputs, max_new_tokens=512, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)
            decoded = decoded[0].split("[/INST]")[1].removesuffix("</s>").strip()
            try:
                # decode the generated ids if json format is well-formed.
                qa = json.loads(decoded)

            except Exception as e:
                print("THERE WAS AN ERROR!!!!!!!!")
                print(decoded)
                print(e)
                continue

            if qa.keys() != {"Q", "A"}:
                print("THERE WAS AN ERROR!!!!!!!!")
                print(decoded)
                continue

            example["qa"].append(qa)

        return example

    mimic_qa = mimic_qa.map(generate_qa)
    mimic_qa.to_json("mimic_qa.json")


if __name__ == "__main__":
    main()
