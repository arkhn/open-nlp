ADAPTER_PATHS := /lora/adapter1,/lora/adapter2
DATASET_PATH := to/dataset/path
DOMAIN := health
ENV_FILE := .env
EVALUATOR_PATH := sentence-transformers-paraphrase-MiniLM-L6-v2
HYDRA_CONFIG := "test"
MODEL_ID := mrx4y3h2
N := 4
PRIVATE_PATH := to/private/path
SIZE := 1500
STEP := sft


jz-download-resources:
	sbatch launch/jz/download_resources.slurm

jz-generation:
	sbatch launch/jz/generation.slurm \
	--DATASET_PATH $(PRIVATE_PATH)/private.parquet \
	--ADAPTERS_PATHS $(ADAPTERS_PATHS) \
	--OUTPUT_PATH datasets/$(DOMAIN)/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)

jz-health-preprocessing:
	sbatch launch/jz/health-preprocessing.slurm

jz-health-evaluation:
	sbatch launch/jz/health-evaluation.slurm \
	--DOWNSTREAM_DS_PATH datasets/health/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)/model=$(EVALUATOR_PATH)_eval.parquet \
	--OUTPUT_PATH datasets/health/eval/model_outputs/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)

jz-score:
	sbatch launch/jz/score.slurm \
	--EVALUATOR_PATH $(EVALUATOR_PATH) \
	--PRIVATE_DATASET $(PRIVATE_PATH)/private.parquet \
	--OUTPUT_PATH datasets/$(DOMAIN)/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP) \
	--N $(N)

jz-dp-sft:
	sbatch launch/jz/dp-sft.slurm --HYDRA_CONFIG $(HYDRA_CONFIG)

jz-dpo:
	sbatch launch/jz/dpo.slurm --HYDRA_CONFIG $(HYDRA_CONFIG)

jz-sft:
	sbatch launch/jz/sft.slurm --HYDRA_CONFIG $(HYDRA_CONFIG)

jz-health-preference-eval:
	sbatch launch/jz/health-preference-eval.slurm --MODEL_ID $(MODEL_ID) --STEP $(STEP) --SIZE $(SIZE)

upload-datasets:
	aws s3 sync ./datasets s3://synth-kg-datasets

download-datasets:
	aws s3 sync s3://synth-kg-datasets ./datasets
