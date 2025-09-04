import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .core.pipeline import Pipeline

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    """Process a batch of document pairs using Hydra configuration"""

    # Hydra automatically sets up output directory and logging
    hydra_cfg = HydraConfig.get()
    log.info(f"Working directory: {hydra_cfg.runtime.output_dir}")
    log.info(f"Processing batch of {cfg.pipeline.dataset_size} document pairs...")

    # Initialize pipeline with config
    pipeline = Pipeline(cfg)

    # Execute pipeline
    result = pipeline.execute(
        dataset_size=cfg.pipeline.dataset_size,
        category_filter=None,
    )

    # Log results
    log.info(f"Total pairs processed: {result['total_pairs']}")
    log.info(f"Successful: {result['successful']}")
    log.info(f"Failed: {result['failed']}")
    log.info(f"Success rate: {result['success_rate']:.1f}%")
    log.info(f"Total processing time: {result['total_processing_time']:.2f}s")

    # Save results
    pipeline.dataset_manager.save_to_json()
    log.info("Results saved to JSON file.")


if __name__ == "__main__":
    main()
