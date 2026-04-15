"""Main SemanticCache class for LLM response caching."""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .embedder import Embedder
from .store import CacheStore
from .utils import format_prompt

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache for LLM responses using similarity-based lookup.
    
    Features:
    - Semantic similarity matching using sentence embeddings
    - Configurable similarity threshold
    - Thread-safe operations
    - Comprehensive logging
    - Statistics tracking
    """
    
    DEFAULT_THRESHOLD = 0.95
    DEFAULT_TOP_K = 1
    
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        embedding_model: Optional[str] = None,
        cache_dir: Optional[str] = None,
        persist: bool = True,
        top_k: int = DEFAULT_TOP_K,
        log_level: int = logging.INFO
    ):
        """
        Initialize the semantic cache.
        
        Args:
            threshold: Similarity threshold (0-1). Higher = stricter matching.
            embedding_model: Name of the sentence-transformers model to use
            cache_dir: Directory for cache persistence
            persist: Whether to persist cache to disk
            top_k: Number of similar entries to check
            log_level: Logging level
        """
        self.threshold = threshold
        self.top_k = top_k
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Initialize embedder (lazy loading)
        self._embedder = Embedder(model_name=embedding_model)
        
        # Initialize store (will be created after embedder loads)
        self._store: Optional[CacheStore] = None
        self._cache_dir = cache_dir
        self._persist = persist
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        self._stats_lock = threading.Lock()
        
        logger.info(f"SemanticCache initialized with threshold={threshold}")
    
    def _setup_logging(self, level: int) -> None:
        """Setup logging configuration."""
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    @property
    def store(self) -> CacheStore:
        """Get or create the cache store."""
        if self._store is None:
            # Initialize store with correct embedding dimension
            self._store = CacheStore(
                embedding_dim=self._embedder.embedding_dim,
                cache_dir=self._cache_dir,
                persist=self._persist
            )
        return self._store
    
    def get(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        system: Optional[str] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Lookup a cached response for a prompt.
        
        Args:
            prompt: The prompt (string or message list)
            system: Optional system message
            **kwargs: Additional parameters to match
            
        Returns:
            Cached response if found and similar enough, None otherwise
        """
        try:
            # Format prompt to canonical string
            prompt_text = format_prompt(prompt, system)
            
            # Generate embedding
            embedding = self._embedder.embed_single(prompt_text)
            
            # Search cache
            results = self.store.search(embedding, k=self.top_k)
            
            if not results:
                logger.debug(f"Cache miss: no entries found")
                self._increment_stat('misses')
                return None
            
            # Check results against threshold
            for entry_id, similarity, response, cached_text in results:
                if similarity >= self.threshold:
                    logger.info(f"Cache hit: similarity={similarity:.4f}, id={entry_id}")
                    self._increment_stat('hits')
                    return response
            
            # No match above threshold
            best_score = results[0][1] if results else 0
            logger.debug(f"Cache miss: best similarity={best_score:.4f} < threshold={self.threshold}")
            self._increment_stat('misses')
            return None
            
        except Exception as e:
            logger.error(f"Error during cache lookup: {e}")
            self._increment_stat('errors')
            return None
    
    def set(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        response: Any,
        system: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """
        Store a response in the cache.
        
        Args:
            prompt: The prompt (string or message list)
            response: The response to cache
            system: Optional system message
            metadata: Optional metadata to store
            **kwargs: Additional parameters (e.g., model) to include in metadata
            
        Returns:
            ID of the cached entry
        """
        try:
            # Format prompt
            prompt_text = format_prompt(prompt, system)
            
            # Generate embedding
            embedding = self._embedder.embed_single(prompt_text)
            
            # Merge kwargs into metadata
            if metadata is None:
                metadata = {}
            metadata.update(kwargs)
            
            # Store in cache
            entry_id = self.store.add(
                embedding=embedding,
                text=prompt_text,
                response=response,
                metadata=metadata
            )
            
            logger.info(f"Cached response: id={entry_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            self._increment_stat('errors')
            raise
    
    def lookup_or_call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        callable_fn: Callable[[], Any],
        system: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Lookup cache or call function if not found.
        
        This is a convenience method that checks the cache first,
        and if not found, calls the provided function and caches the result.
        
        Args:
            prompt: The prompt
            callable_fn: Function to call if cache miss
            system: Optional system message
            metadata: Optional metadata
            **kwargs: Additional parameters
            
        Returns:
            Response (from cache or fresh)
        """
        # Try cache first
        cached = self.get(prompt, system, **kwargs)
        if cached is not None:
            return cached
        
        # Cache miss - call the function
        logger.debug("Cache miss, calling function")
        response = callable_fn()
        
        # Cache the result
        self.set(prompt, response, system, metadata, **kwargs)
        
        return response
    
    def delete(self, entry_id: int) -> bool:
        """
        Delete an entry from the cache.
        
        Args:
            entry_id: Entry ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        return self.store.delete(entry_id)
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self.store.clear()
        with self._stats_lock:
            self._stats = {'hits': 0, 'misses': 0, 'errors': 0}
        logger.info("Cache cleared")
    
    def save(self) -> None:
        """Save cache to disk."""
        self.store.save()
        logger.info("Cache saved")
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        store_stats = self.store.stats()
        
        with self._stats_lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0.0
            
            return {
                **store_stats,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'errors': self._stats['errors'],
                'hit_rate': hit_rate,
                'threshold': self.threshold
            }
    
    def _increment_stat(self, key: str) -> None:
        """Thread-safe stat increment."""
        with self._stats_lock:
            self._stats[key] += 1
    
    def get_similar(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        system: Optional[str] = None,
        k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get similar cached entries without threshold filtering.
        
        Args:
            prompt: The query prompt
            system: Optional system message
            k: Number of results to return
            min_score: Optional minimum similarity score
            
        Returns:
            List of similar entries with scores
        """
        prompt_text = format_prompt(prompt, system)
        embedding = self._embedder.embed_single(prompt_text)
        
        results = self.store.search(embedding, k=k)
        
        entries = []
        for entry_id, similarity, response, cached_text in results:
            if min_score is not None and similarity < min_score:
                continue
            
            entries.append({
                'id': entry_id,
                'similarity': similarity,
                'text': cached_text,
                'response': response
            })
        
        return entries
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save cache."""
        self.save()
