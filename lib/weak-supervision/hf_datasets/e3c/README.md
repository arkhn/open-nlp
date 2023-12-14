# 🐚 E3C Dataset for Transfer Learning, Weak Supervision and Cross-Lingual Explorations

for now, the dataset is located in this huggingface dataset repository: `smeoni/e3c` 🤫 To load the
dataset in a training script, use this following:

```python
from datasets import load_dataset
load_dataset("smeoni/e3c")
```

you can also load the dataset with a local path:

```python
from datasets import load_dataset
load_dataset("./e3c/e3c.py")
```

The README.md inside the e3c folder is the metadata for the dataset. The entire e3c folder in this
repository is uploaded in the huggingface repository. You can generate the metadata using the
following command:

```bash
datasets-cli test e3c/e3c.py --save_infos --all_configs
```
