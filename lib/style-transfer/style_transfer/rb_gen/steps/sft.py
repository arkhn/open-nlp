import hydra
import wandb
from datasets import Dataset
from omegaconf import DictConfig
from peft import PeftModel
from style_transfer.rb_gen.utils.utils import CustomWandbCallback
from trl import SFTTrainer


def sft_train(
    cfg: DictConfig,
    model: PeftModel,
    sft_dataset: Dataset,
    test_dataset: Dataset,
    wandb_log_dict: dict,
) -> PeftModel:
    """Train the model with supervised fine-tuning.

    Args:
        cfg: The configuration for the training.
        model: The model to train.
        sft_dataset: The dataset to use for training.
        test_dataset: The dataset to use for evaluation.
        wandb_log_dict: The dictionary of the dataset sizes.

    Returns:
        The trained model.
    """
    args = hydra.utils.instantiate(cfg.sft.training_args)
    wandb.config.update({"state": "sft"}, allow_val_change=True)
    args.load_best_model_at_end = True
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=sft_dataset,
        eval_dataset=test_dataset,
        callbacks=[CustomWandbCallback],
    )
    trainer.train()
    return model
