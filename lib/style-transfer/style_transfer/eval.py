import json

import hydra
import pandas as pd
import wandb
from omegaconf import omegaconf


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg):
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg,
    )
    with wandb.init(project="eval-style-transfer") as run:
        dataset = run.use_artifact(cfg.dataset)
        json_file = json.load(dataset.files()[0].download(replace=True))
        df = pd.DataFrame(data=json_file["data"], columns=json_file["columns"])
        eval_cols = [f"eval_sem_scores_{i}" for i in range(cfg.num_generated_sequences)]

        df["max_score"] = df[eval_cols].max(axis=1)
        # Determine the best generated text based on the maximum score
        best_generation_indices = df[eval_cols].idxmax(axis=1).apply(lambda x: int(x[-1]))
        df["best_generation"] = best_generation_indices.apply(
            lambda x: df["generation_" + str(x)].iloc[x]
        )

        wandb.log({"test/mean": df["max_score"].mean()})
    wandb.finish()


if __name__ == "__main__":
    main()
