from pathlib import Path
from typing import Dict
import logging


class PromptLoader:
    """
    Utility class for loading and managing prompt templates from external files
    """

    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the PromptLoader

        Args:
            prompts_dir: Directory containing prompt template files
        """
        self.prompts_dir = Path(prompts_dir)
        self._prompt_cache: Dict[str, str] = {}
        self.logger = logging.getLogger(__name__)

        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory '{prompts_dir}' not found")

    def load_prompt(self, prompt_name: str, use_cache: bool = True) -> str:
        """
        Load a prompt template from file

        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            use_cache: Whether to use cached version if available

        Returns:
            Prompt template string

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        # Check cache first
        if use_cache and prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]

        # Construct file path
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"

        if not prompt_file.exists():
            available_prompts = self.list_available_prompts()
            raise FileNotFoundError(
                f"Prompt file '{prompt_file}' not found. " f"Available prompts: {available_prompts}"
            )

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()

            # Cache the loaded prompt
            if use_cache:
                self._prompt_cache[prompt_name] = prompt_content

            self.logger.debug(f"Loaded prompt '{prompt_name}' from {prompt_file}")
            return prompt_content

        except Exception as e:
            self.logger.error(f"Error loading prompt '{prompt_name}': {e}")
            raise

    def list_available_prompts(self) -> list:
        """
        List all available prompt template files

        Returns:
            List of available prompt names (without .txt extension)
        """
        try:
            prompt_files = list(self.prompts_dir.glob("*.txt"))
            return [f.stem for f in prompt_files]
        except Exception as e:
            self.logger.error(f"Error listing prompts: {e}")
            return []

    def reload_prompt(self, prompt_name: str) -> str:
        """
        Force reload a prompt from file, bypassing cache

        Args:
            prompt_name: Name of the prompt to reload

        Returns:
            Reloaded prompt template string
        """
        # Clear from cache first
        if prompt_name in self._prompt_cache:
            del self._prompt_cache[prompt_name]

        return self.load_prompt(prompt_name, use_cache=True)

    def clear_cache(self) -> None:
        """Clear the prompt cache"""
        self._prompt_cache.clear()
        self.logger.debug("Prompt cache cleared")

    def get_prompt_info(self, prompt_name: str) -> Dict[str, str]:
        """
        Get information about a prompt file

        Args:
            prompt_name: Name of the prompt

        Returns:
            Dictionary with prompt file information
        """
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"

        if not prompt_file.exists():
            return {"exists": False, "path": str(prompt_file)}

        try:
            stat = prompt_file.stat()
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "exists": True,
                "path": str(prompt_file),
                "size": stat.st_size,
                "lines": len(content.split("\n")),
                "characters": len(content),
                "in_cache": prompt_name in self._prompt_cache,
            }
        except Exception as e:
            return {"exists": True, "path": str(prompt_file), "error": str(e)}


# Global prompt loader instance
_prompt_loader = None


def get_prompt_loader() -> PromptLoader:
    """
    Get the global PromptLoader instance

    Returns:
        PromptLoader instance
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader


def load_prompt(prompt_name: str) -> str:
    """
    Convenience function to load a prompt using the global loader

    Args:
        prompt_name: Name of the prompt file (without .txt extension)

    Returns:
        Prompt template string
    """
    return get_prompt_loader().load_prompt(prompt_name)
