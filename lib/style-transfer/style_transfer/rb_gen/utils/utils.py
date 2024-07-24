import logging
import os
import tempfile
from pathlib import Path

import datasets
import wandb
from datasets import Dataset, load_dataset
from tokenizers.implementations import BaseTokenizer
from transformers import AutoTokenizer, is_torch_xla_available, modelcard
from transformers.integrations import WandbCallback
from transformers.integrations.integration_utils import save_model_architecture_to_file

logger = logging.getLogger(__name__)


def tokenize(sample: dict, tokenizer: BaseTokenizer, max_sampler_length: int, prompt: str) -> dict:
    """Tokenize the sample.

    Args:
        sample: The sample to tokenize.
        tokenizer: The tokenizer to use.
        max_sampler_length: The maximum length of the input sequence.
        prompt: The prompt to use.

    Returns:
        The tokenized sample.
    """
    ground_ids = tokenizer.encode(sample["text"], add_special_tokens=False)
    ground_ids = (
        ground_ids if len(ground_ids) <= max_sampler_length else ground_ids[:max_sampler_length]
    )
    sample["ground_texts"] = tokenizer.decode(ground_ids)
    sample["keywords"] = ",".join(
        [keyword for keyword in sample["keywords"].split(",") if keyword in sample["ground_texts"]]
    )
    sample["query"] = str.format(prompt, sample["keywords"])
    sample["text"] = sample["query"] + "\n" + sample["text"]
    return sample


def build_dataset(
    dataset_name: str, model_name: str, max_sampler_length: int, prompt: str
) -> Dataset:
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name: The name of the dataset.
        model_name: The name of the model.
        max_sampler_length: The maximum length of the input sequence.
        prompt: The prompt to use.

    Returns:
        The dataset for training / testing.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")

    ds_dict: dict = {"keywords": [], "text": []}
    for keywords, text in zip(ds["keywords"], ds["text"]):
        for kw, t in zip(keywords, text):
            ds_dict["keywords"].append(kw)
            ds_dict["text"].append(t)
    ds = Dataset.from_dict(ds_dict)
    ds = ds.map(
        tokenize,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_sampler_length": max_sampler_length,
            "prompt": prompt,
        },
    )
    ds = ds.filter(lambda x: len(x["keywords"].split(",")) > 1)
    ds.set_format(type="torch")
    return ds


def split_dataset(
    dataset: datasets.Dataset, sft_ratio: float, dpo_ratio: float
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """Split the dataset into train, gen and test.

    Args:
        dataset: The dataset to split.
        sft_ratio: The ratio of the dataset to use for SFT.
        dpo_ratio: The ratio of the dataset to use for DPO.

    Returns:
        The train, gen and test datasets
    """
    # Split the dataset into train, gen and test
    # first we split the dataset into train and test
    sft_dataset, test_dataset = dataset.train_test_split(
        train_size=dpo_ratio, shuffle=False
    ).values()
    # then we split the train dataset into train and gen
    sft_dataset, gen_dataset = sft_dataset.train_test_split(
        train_size=sft_ratio, shuffle=False
    ).values()
    return sft_dataset, gen_dataset, test_dataset


class CustomWandbCallback(WandbCallback):
    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed.
        Find more information [here](https://docs.wandb.ai/guides/integrations/huggingface).
        You can also override the following environment
        variables:

        Environment:
        - **WANDB_LOG_MODEL** (`str`, *optional*, defaults to `"false"`):
            Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"`
            or `"false"`. If set to `"end"`, the model will be uploaded at the end of training.
            If set to `"checkpoint"`, the checkpoint will be uploaded every `args.save_steps` .
            If set to `"false"`, the model will not be uploaded. Use along
            with [`~transformers.TrainingArguments.load_best_model_at_end`] to upload best model.

            <Deprecated version="5.0">

            Setting `WANDB_LOG_MODEL` as `bool` will be deprecated in version 5 of 🤗 Transformers.

            </Deprecated>
        - **WANDB_WATCH** (`str`, *optional* defaults to `"false"`):
            Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`.
            Set to `"all"` to log gradients and parameters.
        - **WANDB_PROJECT** (`str`, *optional*, defaults to `"huggingface"`):
            Set this to a custom string to store results in a different project.
        - **WANDB_DISABLED** (`bool`, *optional*, defaults to `False`):
            Whether to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                "Automatic Weights & Biases "
                'logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = (
                    model.config if isinstance(model.config, dict) else model.config.to_dict()
                )
                combined_dict = {**model_config, **combined_dict}
            if hasattr(model, "peft_config") and model.peft_config is not None:
                peft_config = model.peft_config
                combined_dict = {**{"peft_config": peft_config}, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                init_args["name"] = trial_name
                init_args["group"] = args.run_name
            elif args.run_name is not None:
                init_args["name"] = args.run_name
                if args.run_name == args.output_dir:
                    self._wandb.termwarn(
                        "The `run_name` is currently set to the same value "
                        "as `TrainingArguments.output_dir`. If this was "
                        "not intended, please specify a different run name "
                        "by setting the `TrainingArguments.run_name` parameter.",
                        repeat=False,
                    )

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    **init_args,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update({wandb.config["state"]: combined_dict}, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric(f"{wandb.config['state']}/train/global_step")
                self._wandb.define_metric(
                    "*", step_metric=f"{wandb.config['state']}/train/global_step", step_sync=True
                )

            # keep track of model topology and gradients, unsupported on TPU
            _watch_model = os.getenv("WANDB_WATCH", "false")
            if not is_torch_xla_available() and _watch_model in ("all", "parameters", "gradients"):
                self._wandb.watch(model, log=_watch_model, log_freq=max(100, state.logging_steps))
            self._wandb.run._label(code="transformers_trainer")

            # add number of model parameters to wandb config
            try:
                self._wandb.config["model"]["num_parameters"] = model.num_parameters()
            except AttributeError:
                logger.info("Could not log the number of model parameters in Weights & Biases.")

            # log the initial model architecture to an artifact
            with tempfile.TemporaryDirectory() as temp_dir:
                model_name = (
                    f"model-{self._wandb.run.id}"
                    if (args.run_name is None or args.run_name == args.output_dir)
                    else f"model-{self._wandb.run.name}"
                )
                model_artifact = self._wandb.Artifact(
                    name=model_name,
                    type="model",
                    metadata={
                        "model_config": (
                            model.config.to_dict() if hasattr(model, "config") else None
                        ),
                        "num_parameters": self._wandb.config.get("model").get("num_parameters"),
                        "initial_model": True,
                    },
                )
                # add the architecture to a separate text file
                save_model_architecture_to_file(model, temp_dir)

                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with model_artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(model_artifact, aliases=["base_model"])

                badge_markdown = (
                    f'[<img src="https://raw.githubusercontent.com/wandb/assets/main/'
                    f"wandb-github-badge"
                    f'-28.svg" alt="Visualize in Weights & Biases" width="20'
                    f'0" height="32"/>]({self._wandb.run.get_url()})'
                )

                modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n{badge_markdown}"

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            non_scalar_logs = CustomWandbCallback.rewrite_logs(non_scalar_logs)
            self._wandb.log(
                {**non_scalar_logs, f"{wandb.config['state']}/train/global_step": state.global_step}
            )

    def rewrite_logs(d):
        new_d = {}
        eval_prefix = "eval_"
        eval_prefix_len = len(eval_prefix)
        test_prefix = "test_"
        test_prefix_len = len(test_prefix)
        for k, v in d.items():
            if k.startswith(eval_prefix):
                new_d[wandb.config["state"] + "/eval/" + k[eval_prefix_len:]] = v
            elif k.startswith(test_prefix):
                new_d[wandb.config["state"] + "/test/" + k[test_prefix_len:]] = v
            else:
                new_d[wandb.config["state"] + "/train/" + k] = v
        return new_d
