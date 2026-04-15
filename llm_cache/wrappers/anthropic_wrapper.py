"""Anthropic client wrapper with semantic caching support."""

import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..cache import SemanticCache
from ..utils import format_prompt

logger = logging.getLogger(__name__)


class CachedAnthropic:
    """
    Anthropic client wrapper with semantic caching.
    
    Provides drop-in replacement for Anthropic client with automatic
    semantic caching for messages API.
    """
    
    def __init__(
        self,
        client: Optional[Anthropic] = None,
        cache: Optional[SemanticCache] = None,
        cache_config: Optional[Dict[str, Any]] = None,
        threshold: float = 0.95,
        cache_dir: Optional[str] = None,
        **client_kwargs
    ):
        """
        Initialize cached Anthropic client.

        Args:
            client: Existing Anthropic client (or create new one)
            cache: Existing SemanticCache instance (or create new one)
            cache_config: Configuration for new SemanticCache if not provided
            threshold: Similarity threshold for cache hits (0-1)
            cache_dir: Directory for cache persistence
            **client_kwargs: Arguments to pass to Anthropic client constructor
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required. "
                "Install with: pip install anthropic"
            )

        # Initialize or use provided client
        self._client = client or Anthropic(**client_kwargs)

        # Initialize or use provided cache
        if cache:
            self._cache = cache
        elif cache_config:
            self._cache = SemanticCache(**cache_config)
        else:
            self._cache = SemanticCache(threshold=threshold, cache_dir=cache_dir)

        logger.info("CachedAnthropic initialized")
    
    @property
    def messages(self) -> "CachedMessages":
        """Get cached messages interface."""
        return CachedMessages(self._client.messages, self._cache)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to underlying client."""
        return getattr(self._client, name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class CachedMessages:
    """Cached wrapper for Anthropic messages API."""
    
    def __init__(self, messages: Any, cache: SemanticCache):
        self._messages = messages
        self._cache = cache
    
    def create(
        self,
        model: str,
        max_tokens: int,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Create a message with caching.
        
        Args:
            model: Model name (e.g., "claude-3-opus-20240229")
            max_tokens: Maximum tokens to generate
            messages: List of message dicts with 'role' and 'content'
            system: Optional system prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            Message response (cached or fresh)
        """
        # Check cache first (only for non-streaming)
        if not stream:
            cached = self._cache.get(messages, system=system, model=model, **kwargs)
            if cached is not None:
                logger.info("Cache hit for Anthropic message")
                return cached
        
        # Make API call
        logger.debug("Cache miss, calling Anthropic API")
        response = self._messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            stream=stream,
            **kwargs
        )
        
        # Cache the response (only for non-streaming)
        if not stream:
            self._cache.set(messages, response, system=system, model=model, **kwargs)
        
        return response


class AsyncCachedAnthropic:
    """
    Async Anthropic client wrapper with semantic caching.
    """
    
    def __init__(
        self,
        client: Optional[AsyncAnthropic] = None,
        cache: Optional[SemanticCache] = None,
        cache_config: Optional[Dict[str, Any]] = None,
        threshold: float = 0.95,
        cache_dir: Optional[str] = None,
        **client_kwargs
    ):
        """
        Initialize async cached Anthropic client.

        Args:
            client: Existing AsyncAnthropic client (or create new one)
            cache: Existing SemanticCache instance (or create new one)
            cache_config: Configuration for new SemanticCache if not provided
            threshold: Similarity threshold for cache hits (0-1)
            cache_dir: Directory for cache persistence
            **client_kwargs: Arguments to pass to AsyncAnthropic client constructor
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required. "
                "Install with: pip install anthropic"
            )

        # Initialize or use provided client
        self._client = client or AsyncAnthropic(**client_kwargs)

        # Initialize or use provided cache
        if cache:
            self._cache = cache
        elif cache_config:
            self._cache = SemanticCache(**cache_config)
        else:
            self._cache = SemanticCache(threshold=threshold, cache_dir=cache_dir)

        logger.info("AsyncCachedAnthropic initialized")
    
    @property
    def messages(self) -> "AsyncCachedMessages":
        """Get async cached messages interface."""
        return AsyncCachedMessages(self._client.messages, self._cache)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to underlying client."""
        return getattr(self._client, name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class AsyncCachedMessages:
    """Async cached wrapper for Anthropic messages API."""
    
    def __init__(self, messages: Any, cache: SemanticCache):
        self._messages = messages
        self._cache = cache
    
    async def create(
        self,
        model: str,
        max_tokens: int,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Create an async message with caching.
        
        Args:
            model: Model name (e.g., "claude-3-opus-20240229")
            max_tokens: Maximum tokens to generate
            messages: List of message dicts with 'role' and 'content'
            system: Optional system prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            Message response (cached or fresh)
        """
        # Check cache first (only for non-streaming)
        if not stream:
            cached = self._cache.get(messages, system=system, model=model, **kwargs)
            if cached is not None:
                logger.info("Cache hit for async Anthropic message")
                return cached
        
        # Make API call
        logger.debug("Cache miss, calling Anthropic API")
        response = await self._messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            stream=stream,
            **kwargs
        )
        
        # Cache the response (only for non-streaming)
        if not stream:
            self._cache.set(messages, response, system=system, model=model, **kwargs)
        
        return response
