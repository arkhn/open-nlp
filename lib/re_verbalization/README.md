# RE Verbalization

This experiment is meant to test the verbalization method introduced in
[this paper](https://aclanthology.org/2021.emnlp-main.92.pdf) for few shot RE

# ⚗️ Experiments

## First Experiment

🔗 wandb link: https://wandb.ai/clinical-dream-team/re-verbalization/groups/first_experimentr

🍔 Commands:

```bash
nohup python weak_supervision/train.py -m experiment=first_experiment trainer=gpu
data.fold=0,1,2,3,4 &
```
