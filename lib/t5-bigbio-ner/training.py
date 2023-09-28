import hydra
import wandb
from omegaconf import DictConfig
from src.data import load_preprocess_data
from src.finetuning import finetune_t5
from transformers import T5Tokenizer


@hydra.main(config_name="default", config_path="conf")
def main(config: DictConfig):
    """
    Finetune a T5 model on a dataset

    Args:
        config (OmegaConf): Hydra config
    """
    wandb.init(project=config.wandb.project, entity=config.wandb.entity)
    tokenizer = T5Tokenizer.from_pretrained(config.model.name)
    train_dataset, val_dataset = load_preprocess_data(
        "bio-datasets/bigbio-ner-merged", "instruction", "answer", tokenizer
    )
    finetune_t5(config, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
