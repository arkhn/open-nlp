import logging

import pandas as pd
import plotly.express as px
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Clinical Conflict Data Review",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better healthcare-focused styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #F18F01;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .conflict-highlight {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .document-text {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        height: 300px;
        overflow-y: auto;
        overflow-x: hidden;
        resize: none;
        box-sizing: border-box;
    }
    .document-text::-webkit-scrollbar {
        width: 8px;
    }
    .document-text::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .document-text::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    .document-text::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load and cache the parquet data"""
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")

        # Ensure validation columns exist
        validation_columns = ["validation_status", "confidence_level", "expert_comments"]
        for col in validation_columns:
            if col not in df.columns:
                df[col] = None

        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Return empty DataFrame if file can't be loaded
        return pd.DataFrame()


def save_validation_data(
    df: pd.DataFrame,
    case_id: int,
    validation_status: str,
    confidence_level: int,
    expert_comments: str,
    file_path: str,
):
    """Save validation data to the parquet file"""
    try:
        # Update the specific row
        mask = df["id"] == case_id
        if mask.any():
            df.loc[mask, "validation_status"] = validation_status
            df.loc[mask, "confidence_level"] = confidence_level
            df.loc[mask, "expert_comments"] = (
                expert_comments.strip() if expert_comments and expert_comments.strip() else None
            )

            # Save back to parquet
            df.to_parquet(file_path, index=False)
            logger.info(f"Saved validation data for case {case_id}")
            return True
        else:
            logger.error(f"Case ID {case_id} not found")
            return False
    except Exception as e:
        logger.error(f"Error saving validation data: {e}")
        return False


def format_conflict_type(conflict_type: str) -> str:
    """Format conflict type for better readability"""
    type_mapping = {
        "opposition": "⚔️ Opposition",
        "anatomical": "🦴 Anatomical",
        "value": "📊 Value",
        "contraindication": "⚠️ Contraindication",
        "comparison": "📈 Comparison",
        "descriptive": "📝 Descriptive",
    }
    return type_mapping.get(conflict_type, f"❓ {conflict_type.title()}")


def format_timestamp(timestamp_value) -> str:
    """Safely format timestamp values"""
    if pd.isna(timestamp_value) or timestamp_value is None:
        return "N/A"

    try:
        # If it's already a datetime object
        if hasattr(timestamp_value, "strftime"):
            return timestamp_value.strftime("%Y-%m-%d %H:%M")

        # If it's a string, try to convert to datetime
        if isinstance(timestamp_value, str):
            dt = pd.to_datetime(timestamp_value, errors="coerce")
            if pd.isna(dt):
                return str(timestamp_value)  # Return original string if conversion fails
            return dt.strftime("%Y-%m-%d %H:%M")

        # For other types, try pandas conversion
        dt = pd.to_datetime(timestamp_value, errors="coerce")
        if pd.isna(dt):
            return str(timestamp_value)
        return dt.strftime("%Y-%m-%d %H:%M")

    except Exception:
        return str(timestamp_value)  # Fallback to string representation


def create_conflict_type_chart(df: pd.DataFrame):
    """Create a pie chart of conflict types"""
    conflict_counts = df["conflict_type"].value_counts()

    fig = px.pie(
        values=conflict_counts.values,
        names=[format_conflict_type(t) for t in conflict_counts.index],
        title="Conflict Types Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(font=dict(size=12), showlegend=True, height=400)

    return fig


def create_score_distribution_chart(df: pd.DataFrame):
    """Create a histogram showing the distribution of quality scores"""
    fig = px.histogram(
        df,
        x="score",
        nbins=10,
        title="Quality Score Distribution",
        labels={"score": "Quality Score", "count": "Number of Cases"},
        color_discrete_sequence=["#F18F01"],
        opacity=0.8,
    )

    fig.update_layout(
        xaxis_title="Quality Score (1-5)",
        yaxis_title="Number of Cases",
        height=400,
        showlegend=False,
    )

    # Add vertical line for average score
    avg_score = df["score"].mean()
    fig.add_vline(
        x=avg_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_score:.1f}",
        annotation_position="top",
    )

    return fig


def main():
    """Main application function"""

    # Header
    st.markdown(
        '<h1 class="main-header">🏥 Clinical Conflict Data Review</h1>', unsafe_allow_html=True
    )
    st.markdown(
        """
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        A comprehensive tool for clinical domain experts to validate conflict detection results
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar for file selection and filters
    st.sidebar.markdown("## 📁 Data Source")

    # File upload or selection
    uploaded_file = st.sidebar.file_uploader(
        "Upload Parquet File", type=["parquet"], help="Upload your healthcare conflict data file"
    )

    # Default file path
    default_file = "processed/186fbae0_02092025.parquet"

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_data.parquet", "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_path = "temp_data.parquet"
    else:
        file_path = default_file

    # Load data
    df = load_data(file_path)

    if df.empty:
        st.error("No data loaded. Please check your file path or upload a valid parquet file.")
        st.stop()

    # Sidebar filters
    st.sidebar.markdown("## 🔍 Filters")

    # Conflict type filter
    conflict_types = ["All"] + sorted(df["conflict_type"].unique().tolist())
    selected_conflict_type = st.sidebar.selectbox(
        "Conflict Type", conflict_types, help="Filter by type of conflict detected"
    )

    # Score range filter
    min_score, max_score = st.sidebar.slider(
        "Quality Score Range",
        min_value=int(df["score"].min()),
        max_value=int(df["score"].max()),
        value=(int(df["score"].min()), int(df["score"].max())),
        help="Filter by quality score range",
    )

    # Apply filters
    filtered_df = df.copy()

    if selected_conflict_type != "All":
        filtered_df = filtered_df[filtered_df["conflict_type"] == selected_conflict_type]

    filtered_df = filtered_df[
        (filtered_df["score"] >= min_score) & (filtered_df["score"] <= max_score)
    ]

    # Main content area
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cases", len(filtered_df))

    with col2:
        st.metric("Average Score", f"{filtered_df['score'].mean():.1f}/5")

    with col3:
        st.metric("High Quality Cases", len(filtered_df[filtered_df["score"] >= 4]))

    with col4:
        st.metric("Conflict Types", len(filtered_df["conflict_type"].unique()))

    # Charts section
    st.markdown('<h2 class="section-header">📊 Overview Charts</h2>', unsafe_allow_html=True)

    # Create two equal columns for charts
    col1, col2 = st.columns(2)

    with col1:
        # Conflict type distribution chart
        conflict_chart = create_conflict_type_chart(filtered_df)
        st.plotly_chart(conflict_chart, use_container_width=True)

    with col2:
        # Score distribution chart
        score_chart = create_score_distribution_chart(filtered_df)
        st.plotly_chart(score_chart, use_container_width=True)

    # Detailed case review section
    st.markdown('<h2 class="section-header">🔍 Detailed Case Review</h2>', unsafe_allow_html=True)

    # Case selection
    case_options = []
    for _, row in filtered_df.iterrows():
        validation_indicator = ""
        validation_status = row.get("validation_status")
        if pd.notna(validation_status) and validation_status != "Pending":
            status_emoji = {"Approved": "✅", "Rejected": "❌", "Needs Review": "⚠️"}.get(
                validation_status, "📝"
            )
            validation_indicator = f" {status_emoji}"

        case_options.append(
            f"Case {row['id']} - {format_conflict_type(row['conflict_type'])}\
                 (Score: {row['score']}){validation_indicator}"
        )

    if case_options:
        selected_case_idx = st.selectbox(
            "Select a case to review:",
            range(len(case_options)),
            format_func=lambda x: case_options[x],
            help="Choose a specific case to examine in detail",
        )

        selected_case = filtered_df.iloc[selected_case_idx]

        # Case details
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Case ID:** {selected_case['id']}")
            st.markdown(
                f"**Conflict Type:** {format_conflict_type(selected_case['conflict_type'])}"
            )
            st.markdown(f"**Quality Score:** {selected_case['score']}/5")
            st.markdown(f"**Changes Made:** {selected_case['changes_made']}")

        with col2:
            st.markdown("**Timestamps:**")
            if "doc1_timestamp" in selected_case:
                st.markdown(f"Doc 1: {format_timestamp(selected_case['doc1_timestamp'])}")
            if "doc2_timestamp" in selected_case:
                st.markdown(f"Doc 2: {format_timestamp(selected_case['doc2_timestamp'])}")
            if "created_at" in selected_case:
                st.markdown(f"Created: {format_timestamp(selected_case['created_at'])}")

        # Document comparison
        st.markdown("### 📄 Document Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original Document 1")
            st.markdown(
                f'<div class="document-text">{selected_case["original_doc1_text"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Modified Document 1")
            st.markdown(
                f'<div class="document-text">{selected_case["modified_doc1_text"]}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("#### Original Document 2")
            st.markdown(
                f'<div class="document-text">{selected_case["original_doc2_text"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Modified Document 2")
            st.markdown(
                f'<div class="document-text">{selected_case["modified_doc2_text"]}</div>',
                unsafe_allow_html=True,
            )

        # Conflict highlights
        st.markdown("### ⚠️ Conflict Highlights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original Excerpt 1")
            st.markdown(
                f'<div class="conflict-highlight">{selected_case["original_excerpt_1"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Modified Excerpt 1")
            st.markdown(
                f'<div class="conflict-highlight">{selected_case["modified_excerpt_1"]}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("#### Original Excerpt 2")
            st.markdown(
                f'<div class="conflict-highlight">{selected_case["original_excerpt_2"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Modified Excerpt 2")
            st.markdown(
                f'<div class="conflict-highlight">{selected_case["modified_excerpt_2"]}</div>',
                unsafe_allow_html=True,
            )

        # Validation section
        st.markdown("### ✅ Expert Validation")

        # Load existing validation data
        existing_status = selected_case.get("validation_status", "Pending")
        existing_confidence = selected_case.get("confidence_level", 80)
        existing_comments = selected_case.get("expert_comments", "")

        # Comments section (moved before save button)
        expert_comments = st.text_area(
            "Expert Comments",
            value=existing_comments if pd.notna(existing_comments) else "",
            placeholder="Add your professional assessment and recommendations here...",
            height=100,
            help="Provide detailed comments about the conflict and your validation decision",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            validation_status = st.selectbox(
                "Validation Status",
                ["Pending", "Approved", "Rejected", "Needs Review"],
                index=["Pending", "Approved", "Rejected", "Needs Review"].index(existing_status)
                if existing_status in ["Pending", "Approved", "Rejected", "Needs Review"]
                else 0,
                help="Mark your validation decision for this case",
            )

        with col2:
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0,
                max_value=100,
                value=int(existing_confidence) if pd.notna(existing_confidence) else 80,
                help="Your confidence in this validation (0-100%)",
            )

        with col3:
            if st.button("💾 Save Validation", type="primary"):
                # Save validation data
                success = save_validation_data(
                    df,
                    selected_case["id"],
                    validation_status,
                    confidence_level,
                    expert_comments,
                    file_path,
                )
                if success:
                    st.success(
                        f"Validation saved: {validation_status} (Confidence: {confidence_level}%)"
                    )
                    # Clear cache to reload data
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Failed to save validation data")

    else:
        st.warning("No cases match the current filters. Please adjust your filter settings.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Clinical Conflict Data Review Tool | ARKHN 2025 |
        <a href="#" style="color: #2E86AB;">Help & Documentation</a>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
