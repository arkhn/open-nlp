DATASET_PATH := datasets/health/model=llama-3.3-70b-versatile_t=0.7_size=1500/private.parquet
MODEL_ID := mrx4y3h2
SIZE := 1500
ENV_FILE := .env
PUBLIC_DATASET := path/to/first/dataset
PRIVATE_DATASET := path/to/second/dataset
OUTPUT_PATH := path/to/output/folder
ADAPTER_PATHS := /lora/adapter1 /lora/adapter2
N := 4
HYDRA_CONFIG := ""

.PHONY: sft generation merge-fingpt finance-preprocessing health-preprocessing score alignment health-evaluation finance-evaluation

# Supervised Fine-Tuning
sft:
		sky launch -c sft -y --env-file .env --env HYDRA_CONFIG=$(HYDRA_CONFIG) sft/train.yaml
		sky down -y sft

# Generate synthetic data using a specific model
generation:
		# Launch cluster for data generation
		sky launch -c $(MODEL_ID) -y \
			--env DATASET=$(DATASET_PATH) \
			--env ADAPTERS_PATHS=$(ADAPTERS_PATHS) \
			--env OUTPUT_PATH=$(OUTPUT_PATH) \
			--env-file $(ENV_FILE) \
			launch/generation.yaml

		# Prepare local output directory
		mkdir -p $(OUTPUT_PATH)

		# Download generated data from cluster
		scp $(MODEL_ID):~/sky_workdir/$(OUTPUT_PATH)/public_generated.parquet $(OUTPUT_PATH)/public_generated.parquet

		# Cleanup cluster
		sky down -y $(MODEL_ID)

# Merge FinGPT data
merge-fingpt:
		sky launch -c merge-fingpt -y launch/merge-fingpt.yaml
		sky down -y merge-fingpt

# Data preprocessing tasks
finance-preprocessing:
		python datasets/preprocessing/finance/run.py

health-preprocessing:
		python datasets/preprocessing/health/run.py

# Scoring Step
score:
		python training_steps/score/run.py \
			--model_path $(MODEL_PATH) \
			--public_dataset $(PUBLIC_DATASET) \
			--private_dataset $(PRIVATE_DATASET) \
			--output_path $(OUTPUT_PATH) \
			--n $(N)

# Alignment Step
alignment:
	sky launch -c dpo -y --env-file $(ENV_FILE) --env HYDRA_CONFIG=$(HYDRA_CONFIG) launch/dpo.yaml
	sky down -y dpo
		sky launch -c dpo -y --env-file $(ENV_FILE) --env HYDRA_CONFIG=$(HYDRA_CONFIG) launch/dpo.yaml
		sky down -y dpo