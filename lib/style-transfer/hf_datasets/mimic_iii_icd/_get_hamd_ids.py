import json

import pandas as pd
import wandb
from tqdm import tqdm

tqdm.pandas()


def load_wandb_ds(ds_name):
    api = wandb.Api()
    test_dataset = api.artifact(ds_name)
    json_file = json.load(test_dataset.files()[0].download(replace=True))
    return pd.DataFrame(data=json_file["data"], columns=json_file["columns"])


def main():
    test_ds = load_wandb_ds("clinical-dream-team/gen-style-transfer/run-o8bvb9xv-test_dataset:v0")
    gen_ds = load_wandb_ds("clinical-dream-team/gen-style-transfer/run-o8bvb9xv-gen_dataset:v0")
    note_events = pd.read_csv("raw/NOTEEVENTS.csv", low_memory=False)
    note_events = note_events[note_events["CATEGORY"] == "Discharge summary"]

    def get_hadm_ids(ground_text):
        matching_rows = note_events[
            note_events["TEXT"].str.contains(ground_text[:100], regex=False)
        ]
        return matching_rows["HADM_ID"].tolist()[0]

    for name, ds in [("test", test_ds), ("gen", gen_ds)]:
        ds["HADM_IDS"] = ds["ground_texts"].progress_apply(get_hadm_ids).astype(int)
        ds[["HADM_IDS", "ground_texts"]].to_csv(f"hadm_ids/{name}.csv", index=False)


if __name__ == "__main__":
    main()
