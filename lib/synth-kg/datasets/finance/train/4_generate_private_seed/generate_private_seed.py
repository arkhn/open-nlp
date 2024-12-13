import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# we use a contextual path only because we deploy the script via skypilot
df = pd.read_parquet("./private_seed_data.parquet")
# Extract the specific column
prompts = df["instruction"]


# Initialize the Groq client
client = Groq()

# Generate responses using llama-3.3-70b-versatile
outputs = []
for prompt in tqdm(prompts, desc="Generating responses"):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048
    )
    outputs.append(response.choices[0].message.content)

print(outputs[0])
df = pd.DataFrame({"instruction": prompts, "response": outputs})

# Generate outputs
df.to_parquet("generated_seed_data.parquet")