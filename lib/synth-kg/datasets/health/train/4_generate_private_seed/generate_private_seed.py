import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()
# we use a contextual path only because we deploy the script via skypilot
df = pd.read_parquet("datasets/health/train/4_generate_private_seed/private_seed_data.parquet")
# Extract the specific column
patient_prompts = df["patient_instruction"]
chatbot_prompts = df["chatbot_instruction"]

# Initialize the Groq client
client = Groq()

# Process patient prompts
patient_outputs = []
for prompt in tqdm(patient_prompts, desc="Processing patient prompts"):
                   completion = client.chat.completions.create(
                       model="llama-3.3-70b-versatile",
                       messages=[{"role": "user", "content": prompt}],
                       temperature=0.7,
                       max_tokens=2048
                   )
                   patient_outputs.append(completion.choices[0].message.content)

replaced_chatbot_prompts = [
                   chatbot_prompt.replace("{question}", patient_output)
                   for chatbot_prompt, patient_output in zip(chatbot_prompts, patient_outputs)
]
print(replaced_chatbot_prompts[0])

# Process chatbot prompts
chatbot_outputs = []
for prompt in tqdm(replaced_chatbot_prompts, desc="Processing chatbot prompts"):
                   completion = client.chat.completions.create(
                       model="llama-3.3-70b-versatile",
                       messages=[{"role": "user", "content": prompt}],
                       temperature=0.7,
                       max_tokens=2048
                   )
                   chatbot_outputs.append(completion.choices[0].message.content)

df = pd.DataFrame({"patient_instruction": patient_prompts,
                                  "chatbot_instruction": replaced_chatbot_prompts,
                                  "patient_response": patient_outputs,
                                  "chatbot_response": chatbot_outputs,
                                  })

# Generate outputs
df.to_parquet("generated_seed_data.parquet")