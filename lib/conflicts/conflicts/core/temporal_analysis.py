"""
Temporal analysis utilities for clinical document conflict generation.
Helps analyze time-based relationships between documents to inform conflict decisions.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analyzes temporal relationships between clinical documents"""

    @staticmethod
    def analyze_temporal_relationship(
        doc1_timestamp: Optional[datetime], doc2_timestamp: Optional[datetime]
    ) -> Dict[str, any]:
        """
        Analyze the temporal relationship between two documents

        Args:
            doc1_timestamp: Timestamp of first document
            doc2_timestamp: Timestamp of second document

        Returns:
            Dictionary with temporal analysis results
        """
        if not doc1_timestamp or not doc2_timestamp:
            return {
                "has_temporal_info": False,
                "temporal_relationship": "unknown",
                "time_difference_hours": None,
                "temporal_context": "No timestamp information available",
            }

        # Calculate time difference
        time_diff = abs((doc2_timestamp - doc1_timestamp).total_seconds() / 3600)

        # Determine temporal relationship
        if doc1_timestamp < doc2_timestamp:
            temporal_relationship = "doc1_before_doc2"
            order_description = "Document 1 was created before Document 2"
        elif doc1_timestamp > doc2_timestamp:
            temporal_relationship = "doc2_before_doc1"
            order_description = "Document 2 was created before Document 1"
        else:
            temporal_relationship = "simultaneous"
            order_description = "Documents were created simultaneously"

        # Categorize time difference
        if time_diff < 1:
            time_category = "within_hour"
            time_context = "Documents created within the same hour"
        elif time_diff < 24:
            time_category = "within_day"
            time_context = "Documents created within the same day"
        elif time_diff < 168:  # 7 days
            time_category = "within_week"
            time_context = "Documents created within the same week"
        elif time_diff < 720:  # 30 days
            time_category = "within_month"
            time_context = "Documents created within the same month"
        else:
            time_category = "long_term"
            time_context = "Documents created with significant time separation"

        return {
            "has_temporal_info": True,
            "temporal_relationship": temporal_relationship,
            "time_difference_hours": time_diff,
            "time_category": time_category,
            "order_description": order_description,
            "time_context": time_context,
            "doc1_timestamp": doc1_timestamp.isoformat(),
            "doc2_timestamp": doc2_timestamp.isoformat(),
        }

    @staticmethod
    def get_temporal_conflict_recommendations(temporal_analysis: Dict[str, any]) -> List[str]:
        """
        Get conflict type recommendations based on temporal analysis

        Args:
            temporal_analysis: Result from analyze_temporal_relationship

        Returns:
            List of recommended conflict types
        """
        if not temporal_analysis.get("has_temporal_info"):
            return ["clinical_history", "temporality"]  # Default recommendations

        recommendations = []
        time_category = temporal_analysis.get("time_category")

        # Time-based conflict recommendations
        if time_category == "within_hour":
            # Documents created close together - likely same clinical encounter
            recommendations.extend(
                [
                    "biological_markers",  # Conflicting measurements in same encounter
                    "care_evolution",  # Different assessments of same event
                    "clinical_history",  # Historical inconsistencies
                ]
            )
        elif time_category == "within_day":
            # Same day - could be different shifts or providers
            recommendations.extend(
                [
                    "biological_markers",  # Conflicting lab values
                    "care_evolution",  # Different clinical assessments
                    "clinical_history",  # Historical inconsistencies
                    "temporality",  # Allergy/characteristic conflicts
                ]
            )
        elif time_category == "within_week":
            # Same week - patient condition changes
            recommendations.extend(
                [
                    "care_evolution",  # Care transition conflicts
                    "biological_markers",  # Changing measurements
                    "clinical_history",  # Historical inconsistencies
                    "temporality",  # Characteristic changes
                ]
            )
        elif time_category == "within_month":
            # Same month - longer-term changes
            recommendations.extend(
                [
                    "care_evolution",  # Care transition conflicts
                    "clinical_history",  # Historical inconsistencies
                    "temporality",  # Characteristic changes over time
                ]
            )
        else:  # long_term
            # Significant time separation
            recommendations.extend(
                [
                    "temporality",  # Long-term characteristic changes
                    "clinical_history",  # Historical inconsistencies
                    "care_evolution",  # Care transition conflicts
                ]
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations

    @staticmethod
    def format_temporal_context_for_prompt(temporal_analysis: Dict[str, any]) -> str:
        """
        Format temporal analysis for inclusion in prompts

        Args:
            temporal_analysis: Result from analyze_temporal_relationship

        Returns:
            Formatted string for prompt inclusion
        """
        if not temporal_analysis.get("has_temporal_info"):
            return "Temporal Information: Not available"

        return f"""Temporal Information:
- Document 1 timestamp: {temporal_analysis['doc1_timestamp']}
- Document 2 timestamp: {temporal_analysis['doc2_timestamp']}
- Time relationship: {temporal_analysis['order_description']}
- Time difference: {temporal_analysis['time_difference_hours']:.1f} hours
- Temporal context: {temporal_analysis['time_context']}"""
