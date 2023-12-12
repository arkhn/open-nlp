import pandas as pd

# Load the CSV file to examine its structure
file_path = '/mnt/data/Style Transfer PPO export Dec 12.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
df.head()

from sklearn.model_selection import train_test_split

# Creating a new DataFrame for model training
df_combined = pd.DataFrame({
    'text': pd.concat([df['response'], df['ground_truth']]),
    'label': ([0] * len(df['response'])) + ([1] * len(df['ground_truth']))  # 0 for fake, 1 for real
})

# Splitting the dataset into training and validation sets
train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)

# Displaying the first few rows of the training and validation dataframes
train_df.head(), val_df.head()


Copy code
# Importing necessary libraries from Hugging Face
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenizing the training and validation datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Preparing the datasets for the model
train_dataset = train_dataset.remove_columns(["text"]).with_format("torch")
val_dataset = val_dataset.remove_columns(["text"]).with_format("torch")

# Displaying the first example of the tokenized training dataset
train_dataset[0]

from transformers import BertTokenizer, BertForSequenceClassification
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

from datasets import Dataset

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()


trainer.evaluate()


model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
