import hydra
import wandb
from datasets import Dataset
from omegaconf import DictConfig, omegaconf
from peft import PeftModel
from transformers.integrations import WandbCallback
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

    class CustomWandbCallback(WandbCallback):
        def setup(self, args, state, model, **kwargs):
            super().setup(args, state, model, **kwargs)
            self.wandb_log_dict = wandb_log_dict
            if state.is_world_process_zero:
                dict_conf = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )
                for k_param, v_params in dict_conf.items():
                    setattr(args, k_param, v_params)
                self._wandb.config.update(args, allow_val_change=True)
                self._wandb.config["dataset/size/sft"] = wandb_log_dict["sft_dataset_size"]
                self._wandb.config["dataset/size/test"] = wandb_log_dict["test_dataset_size"]
                self._wandb.config["dataset/size/gen"] = wandb_log_dict["gen_dataset_size"]

    args.load_best_model_at_end = True
    wandb.init(
        project="style-transfer_sft",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=sft_dataset,
        eval_dataset=test_dataset,
        callbacks=[CustomWandbCallback],
    )
    trainer.train()
    wandb.finish()
    return model
