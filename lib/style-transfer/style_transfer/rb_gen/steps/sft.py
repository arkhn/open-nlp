import logging

import hydra
import peft
import wandb
from datasets import Dataset
from omegaconf import DictConfig, ListConfig
from style_transfer.rb_gen.utils.utils import CustomWandbCallback
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def sft_train(
    cfg: DictConfig,
    sft_dataset: Dataset,
    test_dataset: Dataset,
    current_model_path: str,
) -> str:
    """Train the model with supervised fine-tuning.

    This function loads a pre-trained model, applies supervised fine-tuning using the provided
    dataset, and evaluates the model on a test dataset.

    Args:
        cfg (DictConfig): The configuration for the training, containing hyperparameters
        and settings.
        sft_dataset (Dataset): The dataset to use for supervised fine-tuning.
        test_dataset (Dataset): The dataset to use for evaluation after training.
        current_model_path (str): The path to the pre-trained model to be fine-tuned.

    Returns:
        str: The path to the trained model after supervised fine-tuning.

    Note:
        This function uses the SFTTrainer from the TRL library for supervised fine-tuning.
        It also integrates with Weights & Biases (wandb) for experiment tracking.
    """

    logger.info("🦙 load model ...")
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

    cfg.sft.training_args.output_dir = f"models/{wandb.run.id}/sft"
    args = hydra.utils.instantiate(cfg.sft.training_args)
    wandb.config.update({"state": "sft"}, allow_val_change=True)
    test_sft_dataset = None
    if cfg.dataset.sft_dataset is not None:
        sft_dataset, test_sft_dataset = sft_dataset.train_test_split(
            train_size=0.1, shuffle=False
        ).values()
    args.load_best_model_at_end = True
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=sft_dataset,
        eval_dataset=test_dataset if test_sft_dataset is None else test_sft_dataset,
        callbacks=[CustomWandbCallback],
    )
    trainer.train()
    model.save_pretrained(current_model_path)
    del model
    return current_model_path
