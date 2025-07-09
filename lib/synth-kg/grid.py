import logging
import re
import subprocess
import uuid

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def format_adapters(adapters: list[str]) -> str:
    return f"[{','.join(adapters)}]"


def to_str(adapters_path: list[str]) -> str:
    return f"[{','.join(adapters_path)}]"


def format_hydra_args(dataset: str, group_id: str, adapters: list[str]) -> str:
    """
    Build a properly escaped Hydra override string for dataset, adapters, and group ID.
    Important: dataset path values containing '=' must be escaped or quoted appropriately.
    """
    escape_dataset = dataset.replace("=", "\\=")
    return f"\"group_id={group_id} 'adapters_paths={to_str(adapters)}' 'dataset={escape_dataset}'\""


class JobManager:
    """Handles SLURM job submission and dependency tracking."""

    def __init__(self):
        self.last_job_id = -1

    def submit(self, cmd: str) -> int:
        if self.last_job_id != -1:
            cmd = f"sbatch --dependency=afterok:{self.last_job_id} {cmd}"
        else:
            cmd = f"sbatch {cmd}"
        output = self.run_cmd(cmd)
        self.last_job_id = self.extract_job_id(output)
        return self.last_job_id

    def run_cmd(self, cmd: str) -> str:
        log.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(result.stderr)
            raise RuntimeError(result.stderr)
        return result.stdout.strip()

    def extract_job_id(self, output: str) -> int:
        match = re.search(r"Submitted batch job (\d+)", output)
        if not match:
            raise ValueError("No job ID found")
        return int(match.group(1))


class Pipeline:
    """Orchestrates the full training + evaluation pipeline using SLURM."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.bootstrap = cfg.bootstrap
        self.train_steps = cfg.train_steps.split(" ")
        self.adapters = []
        self.current_step = ""
        self.group_id = ""
        self.run_id = ""
        self.job_mgr = JobManager()
        self.sts_model_path = self.cfg.sts_model.replace("/", "-")

    def base_model_path(self) -> str:
        return (
            f"datasets/{self.cfg.domain}/"
            f"model={self.run_id}_size={self.cfg.size}_step={self.cfg.step}"
        )

    def output_model_path(self, step) -> str:
        return (
            f"datasets/{self.cfg.domain}/" f"model={self.run_id}_size={self.cfg.size}_step={step}"
        )

    def dataset_path(self) -> str:
        return (
            f"{self.base_model_path()}/"
            f"model={self.sts_model_path}_scored_{self.cfg.sorting}.parquet"
        )

    def run_bootstrap(self):
        self.group_id = str(uuid.uuid4())[:7]
        self.model_id = self.group_id
        self.run_id = self.group_id
        script = self.cfg.scripts[self.bootstrap]
        dataset = self.cfg.private_path
        escape_dataset = dataset.replace("=", "\\=")
        hydra_args = f"\"'dataset={escape_dataset}'\""
        cmd = f"--export=ALL,WANDB_RUN_ID={self.group_id} {script} " f"--HYDRA_ARGS {hydra_args}"
        self.job_mgr.submit(cmd)
        self.adapters = [f"lora/{self.bootstrap}/{self.group_id}"]
        self.current_step = self.bootstrap

    def run_train_steps(self):
        dataset = self.dataset_path()
        for _, step in enumerate(self.train_steps):
            script = self.cfg.scripts[step.split("-")[0]]
            self.run_id = str(uuid.uuid4())[:7]
            hydra_args = format_hydra_args(dataset, self.group_id, self.adapters)
            cmd = (
                f"--export=ALL,WANDB_RUN_ID={self.run_id} {script} "
                f"--HYDRA_CONFIG {self.cfg.domain} --HYDRA_ARGS {hydra_args}"
            )
            self.job_mgr.submit(cmd)
            self.adapters.append(f"lora/{step}/{self.run_id}")
            self.current_step = step

    def run_generation(self):
        cmd = (
            f"{self.cfg.scripts.generation} "
            f"--DATASET_PATH {self.cfg.private_path} "
            f"--ADAPTERS_PATHS {','.join(self.adapters)} "
            f"--OUTPUT_PATH {self.output_model_path(self.current_step)}"
        )
        self.job_mgr.submit(cmd)

    def run_score(self):
        cmd = (
            f"{self.cfg.scripts.score} "
            f"--STS_MODEL {self.cfg.sts_model} "
            f"--PRIVATE_DATASET {self.cfg.private_path} "
            f"--OUTPUT_PATH {self.output_model_path(self.current_step)} "
            f"--N 4 --WDB_ID {self.run_id} --GROUP_ID {self.group_id}"
        )
        self.job_mgr.submit(cmd)

    def run_filter(self):
        path = (
            f"{self.output_model_path(self.current_step)}/model={self.sts_model_path}"
            "_scored.parquet"
        )
        self.job_mgr.submit(f"{self.cfg.scripts.filter} --INPUT_FILE {path}")

    def run_eval(self):
        path = (
            f"{self.output_model_path(self.current_step)}/model={self.sts_model_path}",
            "_scored_eval.parquet",
        )
        out_path = (
            f"datasets/health/eval/model_outputs/"
            f"model={self.run_id}_size={self.cfg.size}_step={self.current_step}"
        )
        eval_gen_cmd = (
            f"{self.cfg.scripts.eval_gen} --DOWNSTREAM_DS_PATH {path} "
            f"--OUTPUT_PATH {out_path} --GROUP_ID {self.group_id}"
        )
        self.job_mgr.submit(eval_gen_cmd)

        eval_pref_cmd = (
            f"{self.cfg.scripts.eval_preference} "
            f"--MODEL_ID {self.run_id} --STEP {self.current_step} "
            f"--SIZE {self.cfg.size} "
            f"--SUFFIX_RUN_NAME {self.current_step}-{self.cfg.sorting} "
            f"--GROUP_ID {self.group_id}"
        )
        self.job_mgr.submit(eval_pref_cmd)


@hydra.main(config_path=".", config_name="grid.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pipeline = Pipeline(cfg)
    if cfg.bootstrap:
        pipeline.run_bootstrap()
        pipeline.run_generation()
        pipeline.run_score()
        pipeline.run_filter()
        pipeline.run_eval()
    pipeline.run_train_steps()
    pipeline.run_generation()
    pipeline.run_score()
    pipeline.run_filter()
    pipeline.run_eval()


if __name__ == "__main__":
    main()
