import json
import logging
import os

import hydra
import peft
from datasets import Dataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from peft import PeftModel
from style_transfer.rb_gen.steps import dpo_train, generate, score, sft_train
from style_transfer.rb_gen.utils import add_prompt, build_dataset, split_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, set_seed

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_START_METHOD"] = "thread"
tqdm.pandas()


@hydra.main(version_base="1.3", config_path="../configs/rb_gen", config_name="train.yaml")
def main(cfg: DictConfig):
    """Main function for training the model.

    Args:
        cfg: The configuration for the model.
    """
    set_seed(cfg.seed)
    logger.info(json.dumps(OmegaConf.to_container(cfg), indent=4))
    logger.info("💽 Building dataset ...")
    gen_dataset, sft_dataset, test_dataset, wandb_log_dict = init_datasets(cfg)

    logger.info("🦙 load model ...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    logger.info("🍃 Bootstrap Model with Supervised Fine-Tuning...")
    current_model_path = "models/sft/best"
    eval_model_path = "models/score/eval"
    model = sft_train(cfg, model, sft_dataset, test_dataset, wandb_log_dict)
    model.save_pretrained(current_model_path)
    del model

    logger.info("Bootstrapping done,  Iterative Reward-based Generation Training begins...")
    for step in range(cfg.max_steps):
        sth_dataset = generate(
            cfg,
            current_model_path,
            tokenizer,
            gen_dataset,
            test_dataset,
        )
        score_dataset = score(
            cfg,
            step,
            True if step == 0 else False,
            sth_dataset,
            checkpoint=eval_model_path,
        )
        current_model_path = dpo_train(cfg, current_model_path, tokenizer, score_dataset)

    logger.info("🎉 Training Done !")
    logger.info("🔍 Evaluating the final model ...")
    sth_dataset = generate(
        cfg,
        current_model_path,
        tokenizer,
        gen_dataset,
        test_dataset,
    )
    score(
        cfg,
        cfg.max_steps,
        False,
        sth_dataset,
        checkpoint=eval_model_path,
    )


def load_model_and_tokenizer(cfg: DictConfig) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load the model and tokenizer.
    We load the model with PEFT and the tokenizer with padding on the left.
    Args:
        cfg: The configuration for the model.

    Returns:
        The model and tokenizer.
    """
    peft_config = hydra.utils.instantiate(cfg.model.peft_config)
    peft_config.target_modules = (
        list(peft_config.target_modules)
        if isinstance(peft_config.target_modules, ListConfig)
        else peft_config.target_modules
    )

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    model = peft.get_peft_model(
        model,
        peft_config,
    )
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


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
    sft_dataset = sft_dataset.map(
        add_prompt,
        batched=False,
    )
    gen_dataset = gen_dataset.map(
        add_prompt,
        batched=False,
    )
    test_dataset = test_dataset.map(
        add_prompt,
        batched=False,
    )
    logger.info(f"💾 SFT: {len(sft_dataset)}, DPO: {len(gen_dataset)} Test: {len(test_dataset)}")
    wandb_log_dict = {
        "sft_dataset_size": len(sft_dataset),
        "gen_dataset_size": len(gen_dataset),
        "test_dataset_size": len(test_dataset),
    }
    return gen_dataset, sft_dataset, test_dataset, wandb_log_dict


if __name__ == "__main__":
    main()
