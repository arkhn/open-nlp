import logging

import mii
import torch
import yaml
from omegaconf import omegaconf
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

with open("configs/default.yaml", "r") as f:
    cfg = omegaconf.OmegaConf.create(yaml.safe_load(f))
with open("configs/score.yaml", "r") as f:
    cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(yaml.safe_load(f)))

model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=cfg.evaluator,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = model.merge_and_unload()
model.save_pretrained("models/evaluator/")
tokenizer = AutoTokenizer.from_pretrained(cfg.model)
tokenizer.save_pretrained("models/evaluator/")
del model
del tokenizer
logging.info("Model + Tokenizer saved at models/evaluator/")
logging.info("Loading model to pipeline 🐉 ...")
client = mii.serve(
    "models/evaluator/",
    tensor_parallel=4,
    deployment_name=cfg.evaluator,
)
logging.info("Model loaded to pipeline ! 🎉")
