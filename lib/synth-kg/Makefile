DATASET_PATH := datasets/health/model=llama-3.3-70b-versatile_t=0.7_size=1500/private_data.parquet
MODEL_ID := mrx4y3h2
SIZE := 1500
ENV_FILE := .env
PUBLIC_DATASET := path/to/first/dataset
PRIVATE_DATASET := path/to/second/dataset
OUTPUT_PATH := path/to/output/folder
ADAPTER_PATHS := /lora/adapter1 /lora/adapter2

 N := 4
HYDRA_CONFIG := ""
.PHONY: sft generation merge-fingpt finance-preprocessing health-preprocessing score alignment

sft:
	sky launch -c sft -y --env-file .env --env HYDRA_CONFIG=$(HYDRA_CONFIG) sft/train.yaml
	sky down -y sft

generation:
	# launch a sky cluster
	sky launch -c $(MODEL_ID) -y --env DATASET=$(DATASET_PATH) \
			   --env ADAPTERS_PATHS=$(ADAPTERS_PATHS) \
			   --env OUTPUT_PATH=$(OUTPUT_PATH) \
			   --env-file $(ENV_FILE) \
			   launch/generation.yaml
	# create a folder to store the generated data
	mkdir -p $(OUTPUT_PATH)
	# download the generated data
	scp $(MODEL_ID):~/sky_workdir/$(OUTPUT_PATH)/public_generated_data.parquet \
		$(OUTPUT_PATH)/public_generated_data.parquet
	# delete the sky cluster
	sky down -y $(MODEL_ID)

merge-fingpt:
	# launch a sky cluster
	sky launch -c merge-fingpt -y launch/merge-fingpt.yaml
	# delete the sky cluster
	sky down -y merge-fingpt

finance-preprocessing:
	python datasets/preprocessing/finance/run.py

health-preprocessing:
	python datasets/preprocessing/health/run.py

score:
	python training_steps/score/run.py \
		--model_path $(MODEL_PATH) \
		--public_dataset $(PUBLIC_DATASET) \
		--private_dataset $(PRIVATE_DATASET) \
		--output_path $(OUTPUT_PATH) \
		--n $(N)

alignment:
	sky launch -c dpo -y --env-file $(ENV_FILE) --env HYDRA_CONFIG=$(HYDRA_CONFIG) launch/dpo.yaml
	sky down -y dpo