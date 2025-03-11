ADAPTER_PATHS := /lora/adapter1 /lora/adapter2
DATASET_PATH := datasets/health/model=llama-3.3-70b-versatile_t=0.7_size=1500/private.parquet
DOMAIN := health
ENV_FILE := .env
EVALUATOR_PATH := sentence-transformers-paraphrase-MiniLM-L6-v2
FINANCE_EVAL_OUTPUT_PATH := datasets/finance/eval/model_outputs/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)
HEALTH_EVAL_OUTPUT_PATH := datasets/health/eval/model_outputs/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)
EVAL_SFT_PATH := datasets/$(DOMAIN)/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)/model=$(EVALUATOR_PATH)_eval.parquet
HYDRA_CONFIG := "test"
MODEL_ID := mrx4y3h2
N := 4
OUTPUT_PATH := datasets/$(DOMAIN)/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)
PRIVATE_PATH := datasets/health/model=llama-3.3-70b-versatile_t=0.7_size=1500/private.parquet
RESULTS := /path/to/results/folder
SIZE := 1500
STEP := sft
WB_RUN_ID := 1234567890

.PHONY: \
	jz-download-resources \
	jz-generation \
	jz-finance-preprocessing \
	log-results \
	sky-alignment \
	sky-dp-sft \
	sky-finance-evaluation \
	sky-finance-preprocessing \
	sky-generation \
	sky-health-evaluation \
	sky-health-preprocessing \
	sky-merge-fingpt \
	sky-mauve-evaluation \
	sky-score \
	sky-sft

jz-download-resources:
sbatch launch/jz/download_resources.slurm

jz-generation:
sbatch launch/jz/generation.slurm \
	--DATASET_PATH $(DATASET_PATH)/private.parquet \
	--ADAPTERS_PATHS $(ADAPTERS_PATHS) \
	--OUTPUT_PATH $(OUTPUT_PATH)

jz-health-preprocessing:
sbatch launch/jz/health-preprocessing.slurm

jz-health-evaluation:
##  TODO: test with double quote fixing
## Launch evaluation cluster
jz-health-evaluation:
sbatch launch/jz/health-evaluation.slurm \
	--HYDRA_CONFIG="$(HYDRA_CONFIG)" \
	--DATASET="$(EVAL_SFT_PATH)" \
	--OUTPUT_PATH="$(HEALTH_EVAL_OUTPUT_PATH)" \
	--SEED_DATASET="$(DATASET_PATH)"

jz-score:
sbatch launch/jz/score.slurm \
	--EVALUATOR_PATH $(EVALUATOR_PATH) \
	--PUBLIC_DATASET $(OUTPUT_PATH)/public_generated.parquet \
	--PRIVATE_DATASET $(PRIVATE_PATH)/private.parquet \
	--OUTPUT_PATH $(OUTPUT_PATH) \
	--N $(N)

jz-dp-sft:
sbatch launch/jz/dp-sft.slurm --HYDRA_CONFIG $(HYDRA_CONFIG)

jz-dpo:
sbatch launch/jz/dpo.slurm --HYDRA_CONFIG $(HYDRA_CONFIG)

jz-sft:
sbatch launch/jz/sft.slurm --HYDRA_CONFIG $(HYDRA_CONFIG)

local-health-evaluation:
bash ./eval/health/evaluation.sh "$(MODEL_ID)" "$(HEALTH_EVAL_OUTPUT_PATH)/evaluation.parquet"

sky-alignment:
sky launch -c dpo -y -r --env-file $(ENV_FILE) --env HYDRA_CONFIG=$(HYDRA_CONFIG) launch/sky/dpo.yaml
sky down -y dpo

sky-dp-sft:
sky launch -c dp-sft -y -r --env-file .env --env HYDRA_CONFIG=$(HYDRA_CONFIG) launch/sky/dp-sft.yaml
sky down -y dp-sft

sky-finance-evaluation:
python eval/finance/join_datasets.py --evaluation $(EVAL_SFT_PATH) --private_seed $(PRIVATE_PATH) --output_path $(FINANCE_EVAL_OUTPUT_PATH)
cd eval/finance/data && python download.py
sky launch -c finance-evaluation -y -r \
	--env-file $(ENV_FILE) \
	--env HYDRA_CONFIG=$(HYDRA_CONFIG) \
	--env FINANCE_EVAL_OUTPUT_PATH=$(FINANCE_EVAL_OUTPUT_PATH) \
	--env OUTPUT_PATH=$(FINANCE_EVAL_OUTPUT_PATH) \
	--retry-until-up \
	launch/sky/finance-evaluation.yaml

# Cleanup and run evaluation script
sky down -y finance-evaluation

sky-generation:
# Launch cluster for data generation
sky launch -c generation -y -r \
	--env DATASET=$(DATASET_PATH)/private.parquet \
	--env ADAPTERS_PATHS=$(ADAPTERS_PATHS) \
	--env OUTPUT_PATH=$(OUTPUT_PATH) \
	--env-file $(ENV_FILE) \
	launch/sky/generation.yaml

# Prepare local output directory
mkdir -p $(OUTPUT_PATH)

# Download generated data from cluster
scp generation:~/sky_workdir/$(OUTPUT_PATH)/public_generated.parquet $(OUTPUT_PATH)/public_generated.parquet

# Cleanup cluster
sky down -y generation

sky-health-evaluation:
## Launch evaluation cluster
sky launch -c health-evaluation -y -r \
	--env-file $(ENV_FILE) \
	--env HYDRA_CONFIG=$(HYDRA_CONFIG) \
	--env DATASET=$(EVAL_SFT_PATH) \
	--env OUTPUT_PATH=$(HEALTH_EVAL_OUTPUT_PATH) \
	--env SEED_DATASET=$(DATASET_PATH) \
	-r launch/sky/health-evaluation.yaml

## Download evaluation results
mkdir -p $(HEALTH_EVAL_OUTPUT_PATH)
scp health-evaluation:~/sky_workdir/$(HEALTH_EVAL_OUTPUT_PATH)/evaluation.parquet $(HEALTH_EVAL_OUTPUT_PATH)/evaluation.parquet
scp health-evaluation:~/sky_workdir/$(HEALTH_EVAL_OUTPUT_PATH)/evaluation_alpacare_sft.parquet $(HEALTH_EVAL_OUTPUT_PATH)/evaluation_alpacare_sft.parquet

## Cleanup and run evaluation script
sky down -y health-evaluation
bash ./eval/health/evaluation.sh $(MODEL_ID) $(HEALTH_EVAL_OUTPUT_PATH)/evaluation.parquet

sky-health-preprocessing-server:
sky launch -c health-preprocessing -r -y launch/sky/health-preprocessing.yaml

sky-health-preprocessing:
python datasets/preprocessing/health/run.py
sky down -y health-preprocessing

sky-merge-fingpt:
sky launch -c merge-fingpt -r -y launch/sky/merge-fingpt.yaml
sky down -y merge-fingpt

sky-mauve-evaluation:
sky launch -c mauve -y -r launch/sky/mauve.yaml
sky down -y mauve

sky-score:
python training_steps/score/run.py \
	--evaluator_path $(EVALUATOR_PATH) \
	--public_dataset $(OUTPUT_PATH)/public_generated.parquet \
	--private_dataset $(PRIVATE_PATH)/private.parquet \
	--output_path $(OUTPUT_PATH) \
	--n $(N)

sky-sft:
sky launch -c sft -y -r --env-file .env --env HYDRA_CONFIG=$(HYDRA_CONFIG) launch/sky/sft.yaml
sky down -y sft

log-results:
python eval/log_results.py --run-id $(WB_RUN_ID) --results $(RESULTS)

upload-datasets:
aws s3 sync ~/Code/open-nlp/lib/synth-kg/datasets s3://synth-kg-datasets
