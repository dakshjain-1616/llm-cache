"""
LLM Cache - Semantic similarity caching for OpenAI and Anthropic LLM clients.

A production-ready Python middleware that provides intelligent caching for
LLM API calls using FAISS and sentence-transformers for semantic similarity matching.

Example usage:
    >>> from llm_cache import SemanticCache, CachedOpenAI
    >>> 
    >>> # Direct cache usage
    >>> cache = SemanticCache(threshold=0.95)
    >>> response = cache.lookup_or_call(
    ...     "What is the capital of France?",
    ...     lambda: openai_client.chat.completions.create(...)
    ... )
    >>> 
    >>> # OpenAI wrapper
    >>> client = CachedOpenAI(api_key="your-key")
    >>> response = client.chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
"""

__version__ = "0.1.0"
__author__ = "LLM Cache Team"

# Core components
from .cache import SemanticCache
from .embedder import Embedder
from .store import CacheStore
from .utils import (
    cosine_similarity,
    deserialize_response,
    extract_response_text,
    format_prompt,
    hash_prompt,
    serialize_response,
)

# Wrappers
from .wrappers import (
    AsyncCachedAnthropic,
    AsyncCachedOpenAI,
    CachedAnthropic,
    CachedOpenAI,
)

__all__ = [
    # Core classes
    "SemanticCache",
    "Embedder",
    "CacheStore",
    # Wrappers
    "CachedOpenAI",
    "AsyncCachedOpenAI",
    "CachedAnthropic",
    "AsyncCachedAnthropic",
    # Utilities
    "format_prompt",
    "hash_prompt",
    "serialize_response",
    "deserialize_response",
    "extract_response_text",
    "cosine_similarity",
    # Metadata
    "__version__",
]
