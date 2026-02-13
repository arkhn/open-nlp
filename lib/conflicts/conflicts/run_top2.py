import logging

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .core.constants import (
    BIOMARKER_MONITORING_CONFLICT_TYPE,
    CLINICAL_HISTORY_CONFLICT_TYPE,
    PRE_POST_CARE_CONFLICT_TYPE,
    TEMPORALITY_CONFLICT_TYPE,
)
from .core.models import DocumentPair
from .core.pipeline import Pipeline

log = logging.getLogger(__name__)

# Mapping from short names in parquet to full constant names
PARQUET_TO_CONSTANT = {
    "biomarker": BIOMARKER_MONITORING_CONFLICT_TYPE,
    "clinical_history": CLINICAL_HISTORY_CONFLICT_TYPE,
    "pre_post_care": PRE_POST_CARE_CONFLICT_TYPE,
    "temporality": TEMPORALITY_CONFLICT_TYPE,
    # Also support full names as-is
    BIOMARKER_MONITORING_CONFLICT_TYPE: BIOMARKER_MONITORING_CONFLICT_TYPE,
    CLINICAL_HISTORY_CONFLICT_TYPE: CLINICAL_HISTORY_CONFLICT_TYPE,
    PRE_POST_CARE_CONFLICT_TYPE: PRE_POST_CARE_CONFLICT_TYPE,
    TEMPORALITY_CONFLICT_TYPE: TEMPORALITY_CONFLICT_TYPE,
}


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    """Process top2 document pairs per conflict type using the matching doctor agent only."""

    hydra_cfg = HydraConfig.get()
    log.info(f"Working directory: {hydra_cfg.runtime.output_dir}")

    # Initialize pipeline (loads all agents)
    pipeline = Pipeline(cfg)

    # Read top2 parquet
    from pathlib import Path

    parquet_path = Path(__file__).parent.parent / "data" / "top2_per_conflict_type.parquet"
    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded {len(df)} rows from {parquet_path}")

    # Group by conflict_type - each group has 2 documents forming a pair
    for parquet_conflict_type, group in df.groupby("conflict_type"):
        # Map parquet short name to full constant name
        conflict_type = PARQUET_TO_CONSTANT.get(str(parquet_conflict_type))
        if conflict_type is None:
            log.error(
                f"Unknown conflict type in parquet: {parquet_conflict_type}. "
                f"Known mappings: {list(PARQUET_TO_CONSTANT.keys())}"
            )
            continue

        if len(group) < 2:
            log.warning(f"Skipping {conflict_type}: only {len(group)} document(s), need 2")
            continue

        doc1 = group.iloc[0]
        doc2 = group.iloc[1]

        document_pair = DocumentPair(
            doc1_id=str(doc1["row_id"]),
            doc2_id=str(doc2["row_id"]),
            doc1_text=doc1["text"],
            doc2_text=doc2["text"],
            subject_id=f"{doc1['subject_id']},{doc2['subject_id']}",
            category1=doc1["category"],
            category2=doc2["category"],
        )

        pair_id = f"{document_pair.doc1_id}_{document_pair.doc2_id}"
        log.info(f"Processing pair {pair_id} for conflict type: {conflict_type}")

        # Step 1: Extract propositions
        propositions1 = pipeline.proposition_agent(document_pair.doc1_text)
        propositions2 = pipeline.proposition_agent(document_pair.doc2_text)
        proposition_result = (propositions1, propositions2)

        log.info(
            f"Extracted {len(propositions1.propositions)} +"
            f" {len(propositions2.propositions)} propositions"
        )

        # Step 2: Run only the matching doctor agent
        doctor_agent = pipeline.doctor_agents[conflict_type]
        attempts, _ = pipeline._process_single_conflict_type(
            conflict_type, doctor_agent, document_pair, proposition_result
        )

        if not attempts:
            log.error(f"No attempts produced for {conflict_type}")
            continue

        # Find best result
        best_result = None
        for attempt in attempts:
            if attempt["validation_result"].is_valid:
                if (
                    best_result is None
                    or attempt["validation_result"].overall_score
                    > best_result["validation_result"].overall_score
                ):
                    best_result = attempt

        final_result = best_result if best_result else attempts[-1]
        attempt_stats = pipeline._calculate_attempt_statistics(attempts)

        # Step 3: Save
        pipeline._save_attempts_to_database(
            pair_id, document_pair, final_result, attempts, attempt_stats
        )

        status = "VALID" if final_result["validation_result"].is_valid else "INVALID"
        score = final_result["validation_result"].overall_score
        log.info(f"Pair {pair_id} [{conflict_type}]: {status} (score={score:.2f})")

    # Save all results to JSON
    pipeline.dataset_manager.save_to_json()
    log.info("All results saved to JSON file.")


if __name__ == "__main__":
    main()
