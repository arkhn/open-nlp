import pandas as pd
from vllm import LLM, SamplingParams

# we use a contextual path only because we deploy the script via skypilot
df = pd.read_parquet("./private_seed_data.parquet")
# Extract the specific column
prompts = df["instruction"]


# Initialize the LLM with your chosen model
llm = LLM(model="xz97/AlpaCare-llama2-13b")

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
df.to_parquet("generated_seed_data.parquet")
outputs = llm.generate(prompts, sampling_params=sampling_params)
df = pd.DataFrame(
    {"instruction": prompts, "response": [output.outputs[0].text for output in outputs]}
)

# Generate outputs
df.to_parquet("generated_seed_data.parquet")
