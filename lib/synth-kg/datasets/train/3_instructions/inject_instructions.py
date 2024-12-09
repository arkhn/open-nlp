import pandas as pd

# Read the parquet file
df = pd.read_parquet("datasets/train/2_keyword/keyword_data.parquet")
seed_df = df[:500].copy()
gen_df = df[500:].copy()

# seed dataset split
seed_df["new_instruction"] = (
    "Below is an instruction that describes a task. Write a response  that appropriately "
    "completes the request. "
    "### Instruction: As a medical professional, please generate a realistic patient "
    "question from a medical chat format."
    "The question must naturally incorporate these sequences of keywords keeping the order:"
    + seed_df["instruction_keywords"].str.join(", ")
    + "\n\nThen, provide a detailed medical response that incorporates "
    "these sequences of keywords keeping the order: "
    + seed_df["response_keywords"].str.join(", ")
    + "\n\nEnsure the response is clear, professional, and appropriate for patient communication."
    + "the format must be in the following format: Patient: <question> ChatDoctor: <response>."
)
seed_df["new_response"] = (
    "Patient: " + seed_df["instruction"] + "\n\n Chatdoctor: " + seed_df["response"]
)
seed_df = seed_df[["new_instruction", "new_response"]]
seed_df.rename(columns={"new_instruction": "instruction", "new_response": "response"}, inplace=True)
output_path = "datasets/train/4_generate_private_seed/private_seed_data.parquet"
seed_df.to_parquet(output_path, index=False)

# gen dataset split
gen_df["instruction_user_question"] = (
    "Below is an instruction that describes a task. Write a response  that appropriately "
    "completes the request."
    "###Instruction: As a medical professional, please generate a realistic "
    "patient question from a medical chat format."
    "The question should naturally incorporates these sequences of keywords keeping the order: "
    + gen_df["instruction_keywords"].str.join(", ")
    + "\n\nthe format must be in the following format: Patient: <question>"
)
gen_df["response_user_question"] = "Patient: " + gen_df["instruction"]
gen_df["instruction_chatbot_response"] = (
    "Below is an instruction that describes a task. Write a response  that appropriately "
    "completes the request."
    "###Instruction: As a medical professional, please provide a response "
    "from this question: {question}."
    " the response provide a detailed medical response that incorporates these keywords: "
    + gen_df["response_keywords"].str.join(", ")
    + "\n\nEnsure the response is clear, professional, and appropriate for patient communication."
    + "the format must be in the following format: ChatDoctor: <response>."
)
gen_df["response_chatbot_response"] = "Response: " + gen_df["response"]

gen_df["new_response"] = (
    "Question: " + gen_df["instruction"] + "\n\n Response: " + gen_df["response"]
)
gen_df = gen_df[
    [
        "instruction_user_question",
        "instruction_chatbot_response",
        "new_response",
        "response_chatbot_response",
        "response_user_question",
    ]
]
gen_df.rename(columns={"new_response": "response"}, inplace=True)
gen_df.to_parquet("datasets/train/3_instructions/gen_data.parquet", index=False)
