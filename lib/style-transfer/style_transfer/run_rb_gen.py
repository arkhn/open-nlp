import json
import logging

import hydra
import peft
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from style_transfer.rb_gen.steps import dpo_train, generate, score, sft_train
from style_transfer.rb_gen.utils import add_prompt, build_dataset, split_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs/rb_gen", config_name="train.yaml")
def main(cfg: DictConfig):
    """Main function for training the model.

    Args:
        cfg: The configuration for the model.
    """
    set_seed(cfg.seed)
    logger.info(json.dumps(OmegaConf.to_container(cfg), indent=4))
    logger.info("💽 Building dataset ...")
    dataset = build_dataset(
        dataset_name=cfg.dataset.name,
        model_name=cfg.model.name,
        max_sampler_length=cfg.model.max_seq_length,
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

    logger.info("🦙 load model ...")
    peft_config = hydra.utils.instantiate(cfg.model.peft_config)
    peft_config.target_modules = list(peft_config.target_modules)
    model = peft.get_peft_model(
        AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            quantization_config=hydra.utils.instantiate(cfg.model.quantization_config),
        ),
        peft_config,
    )
    trainable_params, all_param = model.get_nb_trainable_parameters()

    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("🍃 Bootstrap Model with Supervised Fine-Tuning...")
    model = sft_train(cfg, model, sft_dataset, test_dataset, wandb_log_dict)
    best_model_path = model.save_pretrained("models/sft/best")
    del model

    logger.info("Bootstrapping done,  Iterative Reward-based Generation Training begins...")
    for step in range(cfg.max_steps):
        sth_dataset = generate(
            cfg,
            best_model_path,
            tokenizer,
            gen_dataset,
            test_dataset,
            wandb_log_dict,
        )
        score_dataset = score(eval_model, sth_dataset, step, wandb_log_dict)
        model = dpo_train(model, score_dataset)

    model.save()


if __name__ == "__main__":
    main()
