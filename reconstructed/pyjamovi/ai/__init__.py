"""AI assistant of OpenStat Studio: natural-language requests run as program commands through the Gemini API.

    from pyjamovi.ai import Assistant, GeminiClient
See assistant.py for the processing steps and knowledge.py for what the model is told about the program.
"""
from .assistant import Assistant, summarize_results
from .gemini_client import KEY_ENV, MODEL_ENV, GeminiClient, GeminiError, list_models

__all__ = ["Assistant", "GeminiClient", "GeminiError", "KEY_ENV", "MODEL_ENV", "list_models", "summarize_results"]
