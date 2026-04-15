"""Embedding module using sentence-transformers with LRU cache and lazy loading."""

import hashlib
import logging
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """
    Text embedder using sentence-transformers with LRU cache and lazy loading.
    
    Uses the 'all-MiniLM-L6-v2' model by default for efficient, high-quality embeddings.
    """
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_EMBEDDING_DIM = 384
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_size: int = 1000,
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name for sentence-transformers
            cache_size: Size of the LRU cache for embeddings
            device: Device to run on ('cpu', 'cuda', or None for auto)
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_size = cache_size
        self.device = device
        self.normalize = normalize
        self._model = None
        self._embedding_dim: Optional[int] = None
        
        logger.info(f"Embedder initialized with model: {self.model_name}")
    
    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded. Embedding dimension: {self._embedding_dim}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def model(self):
        """Get the loaded model (lazy loading)."""
        return self._load_model()
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            _ = self.model  # Trigger lazy loading
        return self._embedding_dim or self.DEFAULT_EMBEDDING_DIM
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Numpy array of embeddings (2D if multiple texts, 1D if single)
        """
        if isinstance(texts, str):
            return self._embed_single_cached(texts)
        
        if not texts:
            return np.array([])
        
        # Batch embedding with cache check
        embeddings = []
        for text in texts:
            emb = self._embed_single_cached(text)
            embeddings.append(emb)
        
        return np.stack(embeddings)
    
    @lru_cache(maxsize=1000)
    def _embed_single_cached(self, text: str) -> np.ndarray:
        """
        Embed a single text with LRU caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embed(text)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0 to 1 for normalized vectors)
        """
        emb1 = self.embed_single(text1)
        emb2 = self.embed_single(text2)
        
        # Cosine similarity = dot product for normalized vectors
        return float(np.dot(emb1, emb2))
    
    def clear_cache(self) -> None:
        """Clear the embedding LRU cache."""
        self._embed_single_cached.cache_clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_info(self) -> dict:
        """
        Get information about the embedding cache.
        
        Returns:
            Dictionary with cache statistics
        """
        info = self._embed_single_cached.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize,
            "currsize": info.currsize,
            "hit_rate": info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0
        }
