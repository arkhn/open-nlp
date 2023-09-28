from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments


def finetune_t5(config, train_dataset, val_dataset):
    """
    Finetune a T5 model on a dataset

    Args:
        config (OmegaConf): Hydra config
        train_dataset (datasets.Dataset): Train dataset
        val_dataset (datasets.Dataset): Validation dataset
    """
    model = T5ForConditionalGeneration.from_pretrained(config.model.name)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        logging_dir="./logs",
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        evaluation_strategy=config.training.eval_strategy,
        load_best_model_at_end=True,
        metric_for_best_model=config.training.metric_for_best_model,
        save_total_limit=config.training.save_total_limit,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    model.push_to_hub(config.model.upload_name)
