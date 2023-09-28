# 🐊 Large Language Models as Instructors: A Study on Multilingual Clinical Entity Extraction

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](https://img.shields.io/badge/BioNLP-2023-blue)](https://aclweb.org/aclwiki/BioNLP_Workshop)

</div>

# 👁️ Description

This project is the codebase used for our weak supervision experiments using E3C dataset annotated with InstructGPT-3 and dictionary.

Considering the E3C dataset, we have compared the models trained with both annotations on the whole language in monolingual and multilingual contexts.

# 🚀 Quick start

```bash
poetry install
```

Train model with default configuration

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python weak_supervision/train.py experiment={experiment_name}
```

You can override any parameter from command line like this:

```bash
python weak_supervision/train.py trainer.max_epochs=20 data.batch_size=64
```

To deploy the project run:

```bash
docker build -t weak_supervision .
docker run -v $(pwd):/workspace/project -e WANDB_API_KEY=$WANDB_API_KEY --gpus all -it  --rm weak_supervision zsh
```

# ⚗️ Experiments

here is a description for each experiment consigned in the Makefile. You see the configuration inside
hydra folder `configs/experiment`:

- **layer_2_comparison**: Performance comparison between two encoder models trained with weak supervision dictionary and InstructGPT-3 annotations on layer 2.
- **layer 2 validation comparison**: Same but comparison between manual and InstructGPT-3 annotations on layer 2 subset.
- **layer 2 blended comparison**: Same experience as **layer_2_comparison** but for each dataset we add a slight quantity of manual annotation.
- **layer 2 blended methods**: we experiment different ratio to blend the dictionary and InstrucGPT-3 annotations.
- **layer 2 xlm**: we trained model with all the data available (all the languages are used) for layer 2. We compare with weak supervision dictionary and InstructGPT-3 annotations.
