"""Constants for conflict types and agent specializations"""

# Specialized conflict types
PRE_POST_CARE_CONFLICT_TYPE = "pre_post_care_evolution"
TEMPORALITY_CONFLICT_TYPE = "temporality"
CLINICAL_HISTORY_CONFLICT_TYPE = "clinical_history_antecedents"
BIOMARKER_MONITORING_CONFLICT_TYPE = "biological_marker_monitoring"

# List of all specialized conflict types
SPECIALIZED_CONFLICT_TYPES = [
    PRE_POST_CARE_CONFLICT_TYPE,
    TEMPORALITY_CONFLICT_TYPE,
    CLINICAL_HISTORY_CONFLICT_TYPE,
    BIOMARKER_MONITORING_CONFLICT_TYPE,
]

# Magic numbers for excerpts
FIRST_EXCERPT = 1
SECOND_EXCERPT = 2
EXCERPT_NUMBERS = [FIRST_EXCERPT, SECOND_EXCERPT]
