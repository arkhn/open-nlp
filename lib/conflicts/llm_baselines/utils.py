"""Shared OpenAI-compatible client and model inference utilities."""

import logging
import os

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv()

log = logging.getLogger(__name__)


def make_client(model_cfg: DictConfig) -> openai.OpenAI:
    """Create an OpenAI-compatible client for a given model config.

    Args:
        model_cfg: Hydra config node with fields: api_key_env, base_url.

    Returns:
        Configured openai.OpenAI client.
    """
    api_key = os.getenv(model_cfg.get("api_key_env", "API_KEY"), "")
    base_url = model_cfg.get("base_url", os.getenv("BASE_URL", ""))
    return openai.OpenAI(api_key=api_key, base_url=base_url or None)


def call_model(
    client: openai.OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> tuple[str, bool]:
    """Call the model and return (response_text, success).

    Args:
        client:        OpenAI-compatible client.
        model_name:    Model identifier string.
        system_prompt: System message content.
        user_prompt:   User message content.
        temperature:   Sampling temperature (0 for deterministic).
        max_tokens:    Maximum tokens in response.

    Returns:
        Tuple of (response text, success flag).
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip(), True
    except Exception as e:
        log.warning(f"Model call failed: {e}")
        return "", False
