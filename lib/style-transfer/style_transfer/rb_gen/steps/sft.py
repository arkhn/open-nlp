import os

import hydra
from omegaconf import omegaconf
from transformers.integrations import WandbCallback
from trl import SFTTrainer

os.environ["WANDB_PROJECT"] = "sft-style-transfer"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_START_METHOD"] = "thread"


@hydra.main(version_base="1.3", config_path="../configs", config_name="sft.yaml")
def sft_train(cfg, model, sft_dataset, test_dataset, wandb_log_dict):
    args = hydra.utils.instantiate(cfg.training_args)

    class CustomWandbCallback(WandbCallback):
        def setup(self, args, state, model, **kwargs):
            super().setup(args, state, model, **kwargs)
            if state.is_world_process_zero:
                dict_conf = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )
                for k_param, v_params in dict_conf.items():
                    setattr(args, k_param, v_params)
                self._wandb.config.update(args, allow_val_change=True)
                self._wandb.config["sft_dataset_size"] = len(sft_dataset)
                self._wandb.config["test_dataset_size"] = len(test_dataset)

    args.run_name = f"sft-ratio-{cfg.sft_ratio}_gen-ratio-{cfg.gen_ratio}"
    args.load_best_model_at_end = True
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=sft_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=True,
        callbacks=[CustomWandbCallback],
    )
    trainer.train()
    return model


if __name__ == "__main__":
    sft_train()
