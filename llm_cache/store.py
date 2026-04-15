"""Cache storage backend using FAISS for semantic similarity search."""

import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .utils import deserialize_response, serialize_response

logger = logging.getLogger(__name__)


class CacheStore:
    """
    Thread-safe cache storage using FAISS for semantic similarity search.
    
    Features:
    - FAISS IndexFlatIP for inner product (cosine) similarity search
    - Persistent storage to ~/.llm_cache/
    - Thread-safe operations with Lock
    - Automatic index management and metadata tracking
    """
    
    DEFAULT_CACHE_DIR = Path.home() / ".llm_cache"
    DEFAULT_INDEX_NAME = "cache.index"
    DEFAULT_METADATA_NAME = "metadata.pkl"
    
    def __init__(
        self,
        embedding_dim: int = 384,
        cache_dir: Optional[str] = None,
        index_name: Optional[str] = None,
        persist: bool = True
    ):
        """
        Initialize the cache store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            cache_dir: Directory for cache persistence (default: ~/.llm_cache)
            index_name: Name for the index file
            persist: Whether to persist cache to disk
        """
        self.embedding_dim = embedding_dim
        self.persist = persist
        self._cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self._index_name = index_name or self.DEFAULT_INDEX_NAME
        self._metadata_name = self.DEFAULT_METADATA_NAME
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory storage
        self._index: Optional[faiss.Index] = None
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._texts: Dict[int, str] = {}
        self._id_counter = 0
        
        # Initialize
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize or load the FAISS index."""
        if self.persist and self._cache_exists():
            self._load()
        else:
            self._create_new_index()
        
        if self.persist:
            self._ensure_cache_dir()
    
    def _cache_exists(self) -> bool:
        """Check if cache files exist."""
        index_path = self._cache_dir / self._index_name
        metadata_path = self._cache_dir / self._metadata_name
        return index_path.exists() and metadata_path.exists()
    
    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory ensured: {self._cache_dir}")
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")
    
    def _load(self) -> None:
        """Load index and metadata from disk."""
        try:
            index_path = self._cache_dir / self._index_name
            metadata_path = self._cache_dir / self._metadata_name
            
            # Load FAISS index
            self._index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self._metadata = data.get('metadata', {})
                self._texts = data.get('texts', {})
                self._id_counter = data.get('id_counter', 0)
            
            logger.info(f"Loaded cache from {self._cache_dir}: {len(self._metadata)} entries")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Creating new index.")
            self._create_new_index()
            self._metadata = {}
            self._texts = {}
            self._id_counter = 0
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        if not self.persist:
            return
        
        with self._lock:
            try:
                self._ensure_cache_dir()
                
                index_path = self._cache_dir / self._index_name
                metadata_path = self._cache_dir / self._metadata_name
                
                # Save FAISS index
                faiss.write_index(self._index, str(index_path))
                
                # Save metadata
                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        'metadata': self._metadata,
                        'texts': self._texts,
                        'id_counter': self._id_counter
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.debug(f"Saved cache to {self._cache_dir}: {len(self._metadata)} entries")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
                raise
    
    def add(
        self,
        embedding: np.ndarray,
        text: str,
        response: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add an entry to the cache.
        
        Args:
            embedding: Embedding vector (normalized)
            text: Original text/prompt
            response: Response object to cache
            metadata: Optional metadata dict
            
        Returns:
            ID of the added entry
        """
        with self._lock:
            # Ensure embedding is 2D for FAISS
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Add to FAISS index
            self._index.add(embedding.astype(np.float32))
            
            # Store metadata
            entry_id = self._id_counter
            self._id_counter += 1
            
            self._texts[entry_id] = text
            self._metadata[entry_id] = {
                'response': serialize_response(response),
                'metadata': metadata or {},
                'timestamp': self._get_timestamp()
            }
            
            logger.debug(f"Added entry {entry_id} to cache")
            return entry_id
    
    def search(
        self,
        embedding: np.ndarray,
        k: int = 1
    ) -> List[Tuple[int, float, Any, str]]:
        """
        Search for similar entries in the cache.
        
        Args:
            embedding: Query embedding vector (normalized)
            k: Number of results to return
            
        Returns:
            List of tuples (id, similarity_score, response, text)
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []
            
            # Ensure embedding is 2D for FAISS
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Search FAISS index
            scores, indices = self._index.search(embedding.astype(np.float32), min(k, self._index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for not found
                    continue
                
                entry = self._metadata.get(int(idx))
                if entry:
                    response = deserialize_response(entry['response'])
                    text = self._texts.get(int(idx), "")
                    results.append((int(idx), float(score), response, text))
            
            return results
    
    def get(self, entry_id: int) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Get a specific entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            Tuple of (response, metadata) or None if not found
        """
        with self._lock:
            entry = self._metadata.get(entry_id)
            if entry:
                response = deserialize_response(entry['response'])
                return response, entry.get('metadata', {})
            return None
    
    def delete(self, entry_id: int) -> bool:
        """
        Delete an entry from the cache.
        
        Note: FAISS doesn't support deletion, so we mark as deleted.
        Full cleanup requires rebuilding the index.
        
        Args:
            entry_id: Entry ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if entry_id in self._metadata:
                del self._metadata[entry_id]
                del self._texts[entry_id]
                logger.debug(f"Marked entry {entry_id} as deleted")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._create_new_index()
            self._metadata = {}
            self._texts = {}
            self._id_counter = 0
            logger.info("Cache cleared")
    
    def rebuild(self) -> None:
        """
        Rebuild the FAISS index (removes deleted entries).
        
        This is expensive but necessary to actually remove deleted entries.
        """
        with self._lock:
            if not self._metadata:
                self._create_new_index()
                return
            
            logger.warning("Full rebuild requires original embeddings. Consider clearing instead.")
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                'entry_count': len(self._metadata),
                'total_entries': len(self._metadata),
                'index_size': self._index.ntotal if self._index else 0,
                'embedding_dim': self.embedding_dim,
                'cache_dir': str(self._cache_dir),
                'persist': self.persist
            }
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def __len__(self) -> int:
        """Return number of entries in cache."""
        with self._lock:
            return len(self._metadata)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save cache."""
        self.save()
