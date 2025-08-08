# Prompt Templates

This directory contains the prompt templates used by the three agents in the clinical conflict
generation pipeline.

## Overview

Instead of storing large text blocks as global variables in `config.py`, prompts are now stored as
external `.txt` files. This provides several benefits:

- **Maintainability**: Easy to edit and update prompts without touching Python code
- **Version Control**: Better tracking of prompt changes in git
- **Readability**: Clean separation of code and prompt content
- **Performance**: Caching system for efficient loading
- **Validation**: Error handling and file validation

## Prompt Files

### `doctor_agent.txt`

- **Purpose**: Analyzes clinical document pairs to determine optimal conflict types
- **Variables**: `{conflict_types}`, `{document1}`, `{document2}`
- **Output**: JSON with conflict_type, reasoning, and modification_instructions

### `editor_agent.txt`

- **Purpose**: Modifies documents to introduce specified conflicts
- **Variables**: `{conflict_type}`, `{conflict_description}`, `{modification_instructions}`,
  `{document1}`, `{document2}`
- **Output**: JSON with modified_document1, modified_document2, and changes_made

### `moderator_agent.txt`

- **Purpose**: Validates modifications for quality and realism
- **Variables**: `{original_doc1}`, `{original_doc2}`, `{modified_doc1}`, `{modified_doc2}`,
  `{conflict_type}`, `{changes_made}`
- **Output**: JSON with is_valid, validation_score, feedback, issues_found, and approval_reasoning

## Usage

### Loading Prompts in Code

```python
from prompt_loader import load_prompt

# Simple loading
doctor_prompt = load_prompt("doctor_agent")

# With formatting
formatted_prompt = doctor_prompt.format(
    conflict_types="Available types...",
    document1="Clinical document 1",
    document2="Clinical document 2"
)
```

### Advanced Usage

```python
from prompt_loader import PromptLoader

# Create loader instance
loader = PromptLoader()

# List available prompts
prompts = loader.list_available_prompts()

# Get detailed info about a prompt
info = loader.get_prompt_info("doctor_agent")

# Force reload from file (bypasses cache)
fresh_prompt = loader.reload_prompt("doctor_agent")

# Clear cache
loader.clear_cache()
```

## File Format

Prompt files use simple text format with Python string formatting placeholders:

```
You are a clinical expert...

Available conflict types:
{conflict_types}

Document 1:
{document1}

Respond with JSON format:
{{
    "field": "value"
}}
```

**Note**: Use double curly braces `{{}}` for literal braces in JSON examples.

## Error Handling

The prompt loader includes comprehensive error handling:

- **Missing files**: Clear error messages with available prompt list
- **Malformed prompts**: Validation during loading
- **Cache management**: Automatic cache invalidation on errors
- **File permissions**: Proper handling of read errors

## Testing

Run the prompt system tests:

```bash
python test_prompts.py
```

This will verify:

- All prompt files are readable
- Formatting variables work correctly
- Caching system functions properly
- Error handling is robust

## Configuration

The prompt system is configured in `config.py`:

```python
# Prompt Configuration
PROMPTS_DIR = "prompts"

# Prompt file names (without .txt extension)
DOCTOR_AGENT_PROMPT_FILE = "doctor_agent"
EDITOR_AGENT_PROMPT_FILE = "editor_agent"
MODERATOR_AGENT_PROMPT_FILE = "moderator_agent"
```

## Best Practices

### Editing Prompts

1. **Test changes**: Use `test_prompts.py` after modifications
2. **Version control**: Commit prompt changes separately from code
3. **Backup**: Keep backups of working prompts before major changes
4. **Validation**: Ensure all placeholder variables are present

### Adding New Prompts

1. Create `.txt` file in `prompts/` directory
2. Add configuration constant in `config.py`
3. Update agents to use the new prompt
4. Add tests for the new prompt

### Prompt Engineering

1. **Clear instructions**: Be specific about expected behavior
2. **Examples**: Include concrete examples when helpful
3. **Format specification**: Clearly define expected output format
4. **Edge cases**: Handle unusual inputs gracefully
5. **Context**: Provide sufficient context for decision making

## Performance Notes

- **Caching**: Prompts are cached after first load for better performance
- **Memory usage**: Minimal memory footprint with on-demand loading
- **File I/O**: Optimized with proper error handling and encoding
- **Thread safety**: Safe for concurrent access (read-only operations)

## Migration from Global Variables

The old system used large string variables in `config.py`:

```python
# OLD (config.py)
DOCTOR_AGENT_PROMPT = """
Large text block here...
"""

# NEW (anywhere in code)
from prompt_loader import load_prompt
doctor_prompt = load_prompt("doctor_agent")
```

This change provides better maintainability and cleaner code structure while maintaining full
backward compatibility in functionality.
