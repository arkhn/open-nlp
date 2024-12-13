import pandas as pd

# Read the input parquet file and split into seed/gen datasets
df = pd.read_parquet("datasets/health/train/2_keyword/keyword_data.parquet")
seed_df = df[:500].copy()
gen_df = df[500:].copy()

# Load prompt templates
patient_prompt = open("datasets/health/train/3_instructions/prompts/patient_instruction.txt").read()
chatbot_prompt = open("datasets/health/train/3_instructions/prompts/chatbot_instruction.txt").read()

# Process seed dataset
seed_df["patient_instruction"] = (
    seed_df["instruction_keywords"]
    .str.join(", ")
    .apply(lambda x: patient_prompt.replace("{keywords}", x))
)
seed_df["chatbot_instruction"] = (
    seed_df["response_keywords"]
    .str.join(", ")
    .apply(lambda x: chatbot_prompt.replace("{keywords}", x))
)
seed_df["patient_response"] = seed_df["instruction"]
seed_df["chatbot_response"] = seed_df["response"]

# Select and rename columns for seed dataset
output_columns = [
    "patient_instruction",
    "chatbot_instruction",
    "patient_response",
    "chatbot_response",
]
seed_df = seed_df[output_columns]
seed_df.rename(columns={"new_response": "response"}, inplace=True)

# Save seed dataset
seed_output_path = "datasets/health/train/4_generate_private_seed/private_seed_data.parquet"
seed_df.to_parquet(seed_output_path, index=False)

gen_df["patient_instruction"] = (
    gen_df["instruction_keywords"]
    .str.join(", ")
    .apply(lambda x: patient_prompt.replace("{kewyords}", x))
)
gen_df["chatbot_instruction"] = (
    gen_df["response_keywords"]
    .str.join(", ")
    .apply(lambda x: chatbot_prompt.replace("{keywords}", x))
)
gen_df["patient_response"] = gen_df["instruction"]
gen_df["chatbot_response"] = gen_df["response"]

# Select and rename columns for gen dataset
gen_df = gen_df[output_columns]
gen_df.rename(columns={"new_response": "response"}, inplace=True)

# Save gen dataset
gen_output_path = "datasets/health/train/3_instructions/gen_data.parquet"
gen_df.to_parquet(gen_output_path, index=False)
