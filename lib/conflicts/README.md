# Clinical Document Conflict Pipeline

A modular Python pipeline that uses the Groq API with LLaMA models to implement a three-agent system for clinical document editing and validation. The system creates conflicts between clinical documents and validates the modifications through an automated workflow.

## System Architecture

### Three-Agent System

1. **Doctor Agent**
   - Analyzes two clinical documents
   - Determines the most appropriate conflict type to introduce
   - Provides specific modification instructions

2. **Editor Agent** 
   - Modifies documents based on Doctor Agent's instructions
   - Creates realistic clinical conflicts while maintaining document integrity
   - Provides detailed change summaries

3. **Moderator Agent**
   - Validates the Editor's modifications
   - Scores the quality and realism of conflicts
   - Approves or rejects modifications

### Pipeline Flow
```
Document Pair → Doctor Agent → Editor Agent → Moderator Agent → Database
                                     ↑              ↓
                                     └── Retry Loop (if invalid) ──┘
```

## Setup

### 1. Prerequisites
- Python 3.8+
- Groq API key
- Clinical document dataset (MIMIC-III or sample data)

### 2. Installation

```bash
# Clone or download the pipeline files
cd clinical-conflicts

# Install dependencies
pip install -r requirements.txt

# Set up your Groq API key
export GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Data Setup

Option A: Use MIMIC-III dataset
```bash
# Run preprocessing script (requires PhysioNet credentials)
python preprocess_dataset.py --user your_username --password your_password
```

Option B: Use sample data (for testing)
```bash
# Sample data will be automatically created when running the pipeline
# if no dataset is found
```

## Usage

### Command Line Interface

The pipeline provides several commands for different use cases:

#### Process Document Batches
```bash
# Process 5 document pairs with default settings
python main.py batch

# Process 10 pairs with custom settings
python main.py batch --size 10 --max-retries 5 --min-score 80

# Process pairs from same subjects only
python main.py batch --size 3 --same-subject

# Filter by document categories
python main.py batch --categories "Discharge summary" "Progress note"
```

#### Test Specific Conflict Types
```bash
# Test opposition conflicts
python main.py test-conflict opposition

# Test anatomical conflicts with more pairs
python main.py test-conflict anatomical --count 5
```

#### View Statistics
```bash
# Show pipeline and dataset statistics
python main.py stats

# Show database information
python main.py db-info

# List all available conflict types
python main.py list-conflicts
```

### Python API

```python
from pipeline import ClinicalConflictPipeline

# Initialize pipeline
pipeline = ClinicalConflictPipeline(
    max_retries=3,
    min_validation_score=70
)

# Process a batch
results = pipeline.process_batch(
    batch_size=5,
    same_subject=False
)

# Get statistics
stats = pipeline.get_pipeline_statistics()
print(f"Validated documents: {stats['validated_documents']}")
```

## Conflict Types

The system supports six types of clinical conflicts:

### 1. Opposition Conflicts
Contradictory findings about the same clinical entity
- Normal vs abnormal findings
- Negative vs positive statements
- Opposite lab interpretations

### 2. Anatomical Conflicts  
Contradictions regarding body structures
- Absent vs present structures
- History of removal vs current presence
- Laterality mismatches

### 3. Value Conflicts
Contradictory measurements and quantitative findings
- Condition vs lab measurement conflicts
- Conflicting lab values at same timestamp

### 4. Contraindication Conflicts
Conflicts between allergies and treatments
- Medication allergies vs prescribed medications

### 5. Comparison Conflicts
Contradictory comparative statements
- Increased/decreased statements vs actual measurements

### 6. Descriptive Conflicts
Contradictory descriptive statements
- Positive vs unlikely statements
- Multiple vs single condition statements

## Configuration

### Environment Variables
```bash
export GROQ_API_KEY="your_api_key"           # Required: Groq API key
```

### Configuration Files

Edit `config.py` to customize:
- API model selection
- Data paths
- Retry limits
- Scoring thresholds
- Prompt templates

## Database Schema

The pipeline uses SQLite to store results:

### validated_documents
- `id`: Primary key
- `original_doc1_text`, `original_doc2_text`: Original documents
- `modified_doc1_text`, `modified_doc2_text`: Modified documents
- `conflict_type`: Type of conflict created
- `validation_score`: Quality score (0-100)
- `changes_made`: Summary of modifications
- `created_at`: Timestamp

### processing_history
- `id`: Primary key
- `doc_pair_id`: Document pair identifier
- `agent_name`: Agent that processed the step
- `result_data`: Processing results (JSON)
- `processing_time`: Time taken
- `created_at`: Timestamp

## Performance

Typical processing times:
- Doctor Agent: 2-5 seconds per pair
- Editor Agent: 3-8 seconds per pair  
- Moderator Agent: 2-4 seconds per pair
- **Total per pair: 7-17 seconds**

Success rates vary by conflict type and dataset quality, typically 60-85%.

## Logging

Logs are written to:
- Console: INFO level and above
- File (`pipeline.log`): All levels
- Database: Processing steps and results

Log levels: DEBUG, INFO, WARNING, ERROR

## Error Handling

The pipeline includes robust error handling:
- API failures: Automatic retry with backoff
- Validation failures: Retry loop up to max attempts
- Data issues: Graceful fallbacks and sample data
- JSON parsing: Error recovery and defaults

## Examples

### Sample Output
```
=== BATCH PROCESSING RESULTS ===
Total pairs processed: 5
Successful: 4
Failed: 1
Success rate: 80.0%
Total processing time: 45.2s
Average per pair: 9.0s
```

### Conflict Example
**Original Document 1:** "Chest X-ray shows no acute cardiopulmonary process"

**Original Document 2:** "Patient has clear lungs bilaterally"

**Modified Document 1:** "Chest X-ray shows bilateral pleural effusions"

**Modified Document 2:** "Patient has clear lungs bilaterally"

**Conflict Type:** Opposition (normal vs abnormal findings)

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   Error: GROQ_API_KEY environment variable is required
   ```
   Solution: Set your Groq API key as environment variable

2. **No Data Found**
   ```
   Warning: Clinical data not found, attempting to create sample data
   ```
   Solution: Either run preprocessing script or use generated sample data

3. **Low Success Rates**
   - Increase max retries: `--max-retries 5`
   - Lower validation threshold: `--min-score 60`
   - Check API connectivity and model availability

4. **API Rate Limits**
   - Add delays between requests
   - Use smaller batch sizes
   - Check Groq API quotas

### Debug Mode
Enable debug logging in `config.py`:
```python
LOG_LEVEL = "DEBUG"
```

## Contributing

To extend the pipeline:

1. **Add new conflict types**: Edit `config.py` CONFLICT_TYPES
2. **Modify agents**: Extend classes in `agents/` directory  
3. **Custom validation**: Override Moderator Agent scoring
4. **New data sources**: Extend ClinicalDataLoader class

## License

This pipeline is for research and educational purposes. Please ensure compliance with data usage agreements when using clinical datasets.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `pipeline.log`
3. Test with sample data first
4. Verify API connectivity
