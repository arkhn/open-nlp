# 🔗 RE Medical Annotations Dataset

HuggingFace Dataset from the Inception Medical Annotations project.

## Data Fields

- `text (str)`: text of the sentence
- `subj_start (int)`: start char of the relation subject mention
- `subj_end (int)`: end char of the relation subject mention, exclusive
- `subj_type (str)`: NER label of the relation subject
- `obj_start (int)`: start char of the relation object mention
- `obj_end (int)`: end char of the relation object mention, exclusive
- `obj_type (str)`: NER label of the relation object
- `relation (str)`: the relation label of this instance

## Usage

This dataset is not pushed on HuggingFace (yet), but it can be used locally with any
archive downloaded from Inception that contains relation annotations.

**Example**: load the dataset from the "RE Temporality POC"

```python
import datasets

ds = datasets.load_dataset(
    "re_medical_annotations/dataset.py",
    data_dir=<Inception Archive path>,
    labels = ["bound"],
)
```
