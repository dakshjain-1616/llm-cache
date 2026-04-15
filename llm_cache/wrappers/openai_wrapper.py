"""OpenAI client wrapper with semantic caching support."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

try:
    import openai
    from openai import AsyncOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..cache import SemanticCache
from ..utils import format_prompt

logger = logging.getLogger(__name__)


class CachedOpenAI:
    """
    OpenAI client wrapper with semantic caching.
    
    Provides drop-in replacement for OpenAI client with automatic
    semantic caching for completions and chat completions.
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        cache: Optional[SemanticCache] = None,
        cache_config: Optional[Dict[str, Any]] = None,
        threshold: float = 0.95,
        cache_dir: Optional[str] = None,
        **client_kwargs
    ):
        """
        Initialize cached OpenAI client.

        Args:
            client: Existing OpenAI client (or create new one)
            cache: Existing SemanticCache instance (or create new one)
            cache_config: Configuration for new SemanticCache if not provided
            threshold: Similarity threshold for cache hits (0-1)
            cache_dir: Directory for cache persistence
            **client_kwargs: Arguments to pass to OpenAI client constructor
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. "
                "Install with: pip install openai"
            )

        # Initialize or use provided client
        self._client = client or OpenAI(**client_kwargs)

        # Initialize or use provided cache
        if cache:
            self._cache = cache
        elif cache_config:
            self._cache = SemanticCache(**cache_config)
        else:
            self._cache = SemanticCache(threshold=threshold, cache_dir=cache_dir)

        logger.info("CachedOpenAI initialized")
    
    @property
    def chat(self) -> "CachedChat":
        """Get cached chat interface (access .completions on the result)."""
        return CachedChat(self._client.chat, self._cache)

    @property
    def completions(self) -> "CachedCompletions":
        """Get cached completions interface."""
        return CachedCompletions(self._client.completions, self._cache)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to underlying client."""
        return getattr(self._client, name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class CachedChat:
    """Intermediate wrapper mirroring OpenAI's client.chat namespace."""

    def __init__(self, chat: Any, cache: SemanticCache):
        self._chat = chat
        self._cache = cache

    @property
    def completions(self) -> "CachedChatCompletions":
        return CachedChatCompletions(self._chat.completions, self._cache)


class CachedChatCompletions:
    """Cached wrapper for OpenAI chat completions."""

    def __init__(self, chat_completions: Any, cache: SemanticCache):
        self._completions = chat_completions
        self._cache = cache
    
    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with caching.
        
        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            Chat completion response (cached or fresh)
        """
        # Check cache first (only for non-streaming)
        if not stream:
            cached = self._cache.get(messages, model=model, **kwargs)
            if cached is not None:
                logger.info("Cache hit for chat completion")
                return cached
        
        # Make API call
        logger.debug("Cache miss, calling OpenAI API")
        response = self._completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
            **kwargs
        )
        
        # Cache the response (only for non-streaming)
        if not stream:
            self._cache.set(messages, response, model=model, **kwargs)
        
        return response


class CachedCompletions:
    """Cached wrapper for OpenAI completions (legacy)."""
    
    def __init__(self, completions: Any, cache: SemanticCache):
        self._completions = completions
        self._cache = cache
    
    def create(
        self,
        model: str,
        prompt: Union[str, List[str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Create a completion with caching.
        
        Args:
            model: Model name
            prompt: Prompt string or list of strings
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            Completion response (cached or fresh)
        """
        # Normalize prompt
        if isinstance(prompt, list):
            prompt_text = "\n".join(prompt)
        else:
            prompt_text = prompt
        
        # Check cache first (only for non-streaming)
        if not stream:
            cached = self._cache.get(prompt_text, model=model, **kwargs)
            if cached is not None:
                logger.info("Cache hit for completion")
                return cached
        
        # Make API call
        logger.debug("Cache miss, calling OpenAI API")
        response = self._completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
            **kwargs
        )
        
        # Cache the response (only for non-streaming)
        if not stream:
            self._cache.set(prompt_text, response, model=model, **kwargs)
        
        return response


class AsyncCachedOpenAI:
    """
    Async OpenAI client wrapper with semantic caching.
    """
    
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        cache: Optional[SemanticCache] = None,
        cache_config: Optional[Dict[str, Any]] = None,
        threshold: float = 0.95,
        cache_dir: Optional[str] = None,
        **client_kwargs
    ):
        """
        Initialize async cached OpenAI client.

        Args:
            client: Existing AsyncOpenAI client (or create new one)
            cache: Existing SemanticCache instance (or create new one)
            cache_config: Configuration for new SemanticCache if not provided
            threshold: Similarity threshold for cache hits (0-1)
            cache_dir: Directory for cache persistence
            **client_kwargs: Arguments to pass to AsyncOpenAI client constructor
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. "
                "Install with: pip install openai"
            )

        # Initialize or use provided client
        self._client = client or AsyncOpenAI(**client_kwargs)

        # Initialize or use provided cache
        if cache:
            self._cache = cache
        elif cache_config:
            self._cache = SemanticCache(**cache_config)
        else:
            self._cache = SemanticCache(threshold=threshold, cache_dir=cache_dir)

        logger.info("AsyncCachedOpenAI initialized")
    
    @property
    def chat(self) -> "AsyncCachedChat":
        """Get async cached chat interface (access .completions on the result)."""
        return AsyncCachedChat(self._client.chat, self._cache)
    
    @property
    def completions(self) -> "AsyncCachedCompletions":
        """Get async cached completions interface."""
        return AsyncCachedCompletions(self._client.completions, self._cache)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to underlying client."""
        return getattr(self._client, name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class AsyncCachedChat:
    """Intermediate async wrapper mirroring OpenAI's client.chat namespace."""

    def __init__(self, chat: Any, cache: SemanticCache):
        self._chat = chat
        self._cache = cache

    @property
    def completions(self) -> "AsyncCachedChatCompletions":
        return AsyncCachedChatCompletions(self._chat.completions, self._cache)


class AsyncCachedChatCompletions:
    """Async cached wrapper for OpenAI chat completions."""

    def __init__(self, chat_completions: Any, cache: SemanticCache):
        self._completions = chat_completions
        self._cache = cache
    
    async def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Create an async chat completion with caching.
        
        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            Chat completion response (cached or fresh)
        """
        # Check cache first (only for non-streaming)
        if not stream:
            cached = self._cache.get(messages, model=model, **kwargs)
            if cached is not None:
                logger.info("Cache hit for async chat completion")
                return cached
        
        # Make API call
        logger.debug("Cache miss, calling OpenAI API")
        response = await self._completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
            **kwargs
        )
        
        # Cache the response (only for non-streaming)
        if not stream:
            self._cache.set(messages, response, model=model, **kwargs)
        
        return response


class AsyncCachedCompletions:
    """Async cached wrapper for OpenAI completions (legacy)."""
    
    def __init__(self, completions: Any, cache: SemanticCache):
        self._completions = completions
        self._cache = cache
    
    async def create(
        self,
        model: str,
        prompt: Union[str, List[str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Create an async completion with caching.
        
        Args:
            model: Model name
            prompt: Prompt string or list of strings
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            Completion response (cached or fresh)
        """
        # Normalize prompt
        if isinstance(prompt, list):
            prompt_text = "\n".join(prompt)
        else:
            prompt_text = prompt
        
        # Check cache first (only for non-streaming)
        if not stream:
            cached = self._cache.get(prompt_text, model=model, **kwargs)
            if cached is not None:
                logger.info("Cache hit for async completion")
                return cached
        
        # Make API call
        logger.debug("Cache miss, calling OpenAI API")
        response = await self._completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
            **kwargs
        )
        
        # Cache the response (only for non-streaming)
        if not stream:
            self._cache.set(prompt_text, response, model=model, **kwargs)
        
        return response
