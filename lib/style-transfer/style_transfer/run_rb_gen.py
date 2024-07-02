import logging

import hydra
from sentence_transformers import SentenceTransformer
from style_transfer.rb_gen.steps import dpo_train, generate, score, sft_train
from style_transfer.rb_gen.utils import add_prompt, build_dataset, split_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs/rb_gen", config_name="train.yaml")
def main(cfg):
    set_seed(cfg.seed)
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

    model = AutoModelForCausalLM.from_pretrained(cfg.model)
    peft_config = hydra.utils.instantiate(cfg.lora)
    peft_config.target_modules = list(peft_config.target_modules)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = sft_train(cfg, model, tokenizer, sft_dataset, gen_dataset, test_dataset, wandb_log_dict)
    eval_model = SentenceTransformer(cfg.sem_model.name)

    for step in range(cfg.max_steps):
        sth_dataset = generate(model, dataset, wandb_log_dict)
        score_dataset = score(eval_model, sth_dataset, step, wandb_log_dict)
        model = dpo_train(model, score_dataset)

    model.save()


if __name__ == "__main__":
    main()
