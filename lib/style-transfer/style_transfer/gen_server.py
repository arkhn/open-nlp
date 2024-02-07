import logging

import mii
import torch
import wandb
import yaml
from omegaconf import omegaconf
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

with open("configs/default.yaml", "r") as f:
    cfg = omegaconf.OmegaConf.create(yaml.safe_load(f))
with open("configs/gen.yaml", "r") as f:
    cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(yaml.safe_load(f)))

api = wandb.Api()
model_artifact = api.artifact(cfg.checkpoint)
model_dir = model_artifact.download()
model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = model.merge_and_unload()
model.save_pretrained("models/merged/")
tokenizer = AutoTokenizer.from_pretrained(cfg.model)
tokenizer.save_pretrained("models/merged/")
del model
del tokenizer
logging.info("Model + Tokenizer saved at models/merged/")
logging.info("Loading model to pipeline 🐉 ...")
client = mii.serve(
    "models/merged/",
    tensor_parallel=4,
    deployment_name=cfg.checkpoint,
)
logging.info("Model loaded to pipeline ! 🎉")
