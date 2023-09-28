import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from style_transfer.train import train


@pytest.mark.slow
def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    Args:
        cfg_train (DictConfig): Configuration composed by Hydra.
    """
    for dataset in ["yelp", "mimic_iii", "pmc_patients"]:
        HydraConfig().set_config(cfg_train)
        with open_dict(cfg_train):
            cfg_train.trainer.fast_dev_run = True
            cfg_train.trainer.accelerator = "cpu"
            cfg_train.datasets = dataset
        train(cfg_train)


@pytest.mark.slow
def test_train_fast_dev_run_ppo(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    Args:
        cfg_train (DictConfig): Configuration composed by Hydra.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.models = "ppo_t5"
    train(cfg_train)
