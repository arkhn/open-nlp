import json
import logging
import os
import shutil

import hydra
import wandb
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf, omegaconf
from style_transfer.rb_gen.steps import dpo_train, generate, score, sft_train
from style_transfer.rb_gen.utils import build_dataset, split_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, set_seed

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_LOG_MODEL"] = "none"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
tqdm.pandas()


@hydra.main(version_base="1.3", config_path="../configs/rb_gen", config_name="train.yaml")
def main(cfg: DictConfig):
    """Main function for training the model.

    Args:
        cfg: The configuration for the model.
    """
    wandb.init(
        project="style-transfer",
        id=f"{cfg.model.name.replace('/', '-')}-{wandb.util.generate_id()}",
        resume="allow",
        settings=wandb.Settings(code_dir="."),
    )
    wandb.config.update(
        omegaconf.OmegaConf.to_container(
            cfg,
        ),
        allow_val_change=True,
    )
    wandb.run.log_code("./style_transfer")
    set_seed(cfg.seed)
    logger.info("wandb_dir: {}".format(wandb.run.dir))
    logger.info(json.dumps(OmegaConf.to_container(cfg), indent=4))
    logger.info("💽 Building dataset ...")
    gen_dataset, sft_dataset, test_dataset, wandb_log_dict = init_datasets(cfg)
    # instead use config update

    wandb.config.update(
        {
            "dataset": {
                "size/sft": wandb_log_dict["sft_dataset_size"],
                "size/test": wandb_log_dict["test_dataset_size"],
                "size/gen": wandb_log_dict["gen_dataset_size"],
                **wandb.config.dataset,
            }
        },
        allow_val_change=True,
    )
    tokenizer = load_tokenizer(cfg)
    logger.info("🍃 Bootstrap Model with Supervised Fine-Tuning...")
    current_model_path = f"models/{wandb.run.id}/sft/best"
    eval_model_path = f"models/{wandb.run.id}/score/eval"
    sft_train(cfg, sft_dataset, test_dataset, current_model_path)
    logger.info("Bootstrapping done,  Iterative Reward-based Generation Training begins...")
    for step in range(cfg.max_steps):
        logger.info(f"🔄 Step {step} ...")
        sth_dataset = generate(
            cfg,
            step,
            current_model_path,
            tokenizer,
            gen_dataset,
        )
        score_dataset = score(
            cfg,
            step,
            True if step == 0 else False,
            sth_dataset,
            checkpoint=eval_model_path,
        )
        current_model_path = dpo_train(cfg, step, current_model_path, tokenizer, score_dataset)

    logger.info("🎉 Training Done !")
    logger.info("🔍 Evaluating the final model ...")
    sth_dataset = generate(
        cfg,
        cfg.max_steps,
        current_model_path,
        tokenizer,
        gen_dataset,
    )
    score(
        cfg,
        cfg.max_steps,
        False,
        sth_dataset,
        checkpoint=eval_model_path,
    )
    shutil.rmtree(f"models/{wandb.run.id}/merged/")
    wandb.finish()


def load_tokenizer(cfg: DictConfig) -> PreTrainedTokenizerBase:
    """Load the model and tokenizer.
    We load the model with PEFT and the tokenizer with padding on the left.
    Args:
        cfg: The configuration for the model.

    Returns:
        The model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def init_datasets(cfg) -> tuple[Dataset, Dataset, Dataset, dict]:
    """Initialize the datasets.
    We build the dataset, split it into sft, gen and test datasets
    and add the prompt to each dataset.
    We preemptively use the max_seq_length from the model configuration to build the dataset using
    the build_dataset function.

    Args:
        cfg: The configuration for the model.

    Returns:
        The gen, sft and test datasets and the wandb log dictionary.
    """

    dataset = build_dataset(
        dataset_name=cfg.dataset.name,
        model_name=cfg.model.name,
        max_sampler_length=cfg.model.max_seq_length,
        prompt=cfg.model.prompt,
    )
    sft_dataset, gen_dataset, test_dataset = split_dataset(
        dataset,
        cfg.dataset.sft_ratio,
        cfg.dataset.gen_ratio,
    )

    if cfg.dataset.sft_dataset is not None:
        sft_dataset = build_dataset(
            dataset_name=cfg.dataset.sft_dataset.name,
            model_name=cfg.model.name,
            max_sampler_length=cfg.model.max_seq_length,
            prompt=cfg.model.prompt,
        )
        sft_dataset = sft_dataset.select(range(cfg.dataset.sft_dataset.size))

    logger.info(f"💾 SFT: {len(sft_dataset)}, DPO: {len(gen_dataset)} Test: {len(test_dataset)}")
    wandb_log_dict = {
        "sft_dataset_size": len(sft_dataset),
        "gen_dataset_size": len(gen_dataset),
        "test_dataset_size": len(test_dataset),
    }
    return gen_dataset, sft_dataset, test_dataset, wandb_log_dict


if __name__ == "__main__":
    main()
