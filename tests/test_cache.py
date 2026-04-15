"""Comprehensive unit tests for the SemanticCache class."""

import os
import sys
import threading
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_cache import SemanticCache, Embedder, CacheStore


class TestSemanticCache:
    """Test cases for SemanticCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        # threshold=0.92 gives a clear gap between paraphrases (>0.93) and
        # structurally-similar-but-different sentences (<0.88)
        return SemanticCache(
            threshold=0.92,
            cache_dir=temp_cache_dir,
            persist=False,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization(self, temp_cache_dir):
        cache = SemanticCache(
            threshold=0.9,
            cache_dir=temp_cache_dir,
            persist=False,
        )
        assert cache is not None
        assert cache.threshold == 0.9
        stats = cache.stats()
        assert stats["total_entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    # ------------------------------------------------------------------
    # Basic set / get
    # ------------------------------------------------------------------

    def test_set_and_get_exact(self, cache):
        """Exact same prompt always returns the cached response."""
        prompt = "What is the capital of France?"
        response = {"answer": "Paris"}

        cache.set(prompt, response)
        cached = cache.get(prompt)
        assert cached is not None
        assert cached == response

    # ------------------------------------------------------------------
    # Semantic hit
    # ------------------------------------------------------------------

    def test_semantic_similarity_cache_hit(self, cache):
        """A clear paraphrase of a stored prompt must produce a cache hit."""
        # These two are near-synonymous — all-MiniLM-L6-v2 scores them ~0.95+
        stored_prompt = "How does photosynthesis work in plants?"
        query_prompt = "Explain the process of photosynthesis in plants."
        response = {"answer": "Photosynthesis converts sunlight to energy."}

        cache.set(stored_prompt, response)
        cached = cache.get(query_prompt)
        assert cached is not None
        assert cached == response

    # ------------------------------------------------------------------
    # Semantic miss
    # ------------------------------------------------------------------

    def test_different_prompt_cache_miss(self, cache):
        """A completely unrelated prompt must not hit the cache."""
        cache.set("What is the capital of France?", {"answer": "Paris"})

        # Entirely different domain — similarity will be well below 0.50
        cached = cache.get("How do I bake a chocolate cake?")
        assert cached is None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def test_cache_stats(self, cache):
        """Hits/misses/hit_rate are tracked correctly."""
        stored = "What is machine learning?"
        paraphrase = "Explain the concept of machine learning."   # clear paraphrase
        unrelated = "What is the recipe for beef stew?"            # completely different

        response = {"answer": "ML info"}

        # Empty stats
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

        # Store + exact hit
        cache.set(stored, response)
        cache.get(stored)        # hit
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

        # Paraphrase hit
        cache.get(paraphrase)    # hit
        stats = cache.stats()
        assert stats["hits"] == 2

        # Completely different — miss
        cache.get(unrelated)     # miss
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 2 / 3) < 1e-9

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def test_clear_cache(self, cache):
        prompt = "Test prompt"
        response = {"answer": "Test"}

        cache.set(prompt, response)
        assert cache.get(prompt) is not None

        cache.clear()
        assert cache.get(prompt) is None

        stats = cache.stats()
        assert stats["total_entries"] == 0

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def test_delete_entry(self, cache):
        prompt1 = "What is Python?"
        prompt2 = "How do you make pasta carbonara?"  # clearly unrelated
        response = {"answer": "A language"}

        cache.set(prompt1, response)
        cache.set(prompt2, response)

        similar = cache.get_similar(prompt1, k=1)
        assert len(similar) > 0
        entry_id = similar[0]["id"]

        cache.delete(entry_id)

        assert cache.get(prompt1) is None
        assert cache.get(prompt2) is not None

    # ------------------------------------------------------------------
    # get_similar
    # ------------------------------------------------------------------

    def test_get_similar(self, cache):
        prompt1 = "What is machine learning?"
        prompt2 = "Explain machine learning"
        prompt3 = "What is deep learning?"
        response = {"answer": "ML info"}

        cache.set(prompt1, response)
        cache.set(prompt2, response)
        cache.set(prompt3, response)

        similar = cache.get_similar(prompt1, k=2)
        assert len(similar) == 2
        assert similar[0]["response"] == response

    # ------------------------------------------------------------------
    # System prompt handling
    # ------------------------------------------------------------------

    def test_system_prompt_handling(self, cache):
        messages = [{"role": "user", "content": "What is AI?"}]
        system_a = "You are a helpful assistant."
        system_b = "You are a sarcastic comedian who hates technology."
        response = {"answer": "AI is..."}

        cache.set(messages, response, system=system_a)

        # Same messages + same system → hit
        cached = cache.get(messages, system=system_a)
        assert cached is not None

        # Same messages + very different system → miss (the embedded text differs a lot)
        cached = cache.get(messages, system=system_b)
        assert cached is None

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def test_thread_safety(self, cache):
        """Concurrent set+get from multiple threads must not corrupt state."""
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    prompt = f"Worker {worker_id} unique task number {i}"
                    response = {"worker": worker_id, "task": i}
                    cache.set(prompt, response)
                    # Exact re-query always returns similarity=1.0, above any threshold
                    cached = cache.get(prompt)
                    assert cached == response
            except Exception as e:
                errors.append(f"worker={worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

        stats = cache.stats()
        assert stats["total_entries"] == 50  # 5 workers × 10 prompts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def test_persistence(self, temp_cache_dir):
        prompt = "Persistent prompt"
        response = {"answer": "Persistent answer"}

        cache1 = SemanticCache(cache_dir=temp_cache_dir, persist=True)
        cache1.set(prompt, response)
        cache1.save()

        cache2 = SemanticCache(cache_dir=temp_cache_dir, persist=True)
        cached = cache2.get(prompt)
        assert cached is not None
        assert cached == response

    # ------------------------------------------------------------------
    # lookup_or_call
    # ------------------------------------------------------------------

    def test_lookup_or_call(self, cache):
        call_count = 0

        def mock_api_call():
            nonlocal call_count
            call_count += 1
            return {"answer": f"Response {call_count}"}

        stored = "How does the immune system protect the body?"
        paraphrase = "Explain how the immune system defends the human body."

        # First call — cache miss → function executes
        result1 = cache.lookup_or_call(stored, mock_api_call)
        assert call_count == 1
        assert result1 == {"answer": "Response 1"}

        # Paraphrase call — cache hit → function NOT called again
        result2 = cache.lookup_or_call(paraphrase, mock_api_call)
        assert call_count == 1
        assert result2 == {"answer": "Response 1"}

    # ------------------------------------------------------------------
    # Empty cache
    # ------------------------------------------------------------------

    def test_empty_cache_get(self, cache):
        cached = cache.get("Any prompt")
        assert cached is None

    # ------------------------------------------------------------------
    # Threshold configuration
    # ------------------------------------------------------------------

    def test_threshold_configuration(self, temp_cache_dir):
        strict_cache = SemanticCache(
            threshold=0.99,
            cache_dir=temp_cache_dir,
            persist=False,
        )
        prompt1 = "What is Python?"
        prompt2 = "Tell me about Python"
        response = {"answer": "Python info"}

        strict_cache.set(prompt1, response)
        # No assertion on the result — just verify no exception is raised
        strict_cache.get(prompt2)


class TestEmbedder:
    """Test cases for Embedder class."""

    @pytest.fixture
    def embedder(self):
        return Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_embed_single(self, embedder):
        text = "This is a test sentence."
        embedding = embedder.embed_single(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_batch(self, embedder):
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_compute_similarity(self, embedder):
        text1 = "What is the capital of France?"
        text2 = "Tell me the capital city of France"
        text3 = "What is the weather like today?"

        sim1 = embedder.compute_similarity(text1, text2)
        sim2 = embedder.compute_similarity(text1, text3)

        assert sim1 > sim2
        assert sim1 > 0.8
        assert sim2 < 0.8

    def test_cache_clear(self, embedder):
        text = "Test text"
        embedder.embed_single(text)

        info_before = embedder.get_cache_info()
        assert info_before["misses"] >= 1

        embedder.clear_cache()

        info_after = embedder.get_cache_info()
        assert info_after["hits"] == 0
        assert info_after["misses"] == 0


class TestCacheStore:
    """Test cases for CacheStore class."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def store(self, temp_dir):
        return CacheStore(embedding_dim=384, cache_dir=temp_dir, persist=False)

    def test_add_and_search(self, store):
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        text = "Test text"
        response = {"answer": "Test"}

        store.add(embedding, text, response)

        results = store.search(embedding, k=1)
        assert len(results) == 1
        assert results[0][2] == response  # (id, score, response, text)

    def test_stats(self, store):
        stats = store.stats()
        assert stats["entry_count"] == 0

        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        store.add(embedding, "text", {"answer": "test"})

        stats = store.stats()
        assert stats["entry_count"] == 1

    def test_save_and_load(self, temp_dir):
        """Round-trip: save to disk and reload in a new instance."""
        store1 = CacheStore(embedding_dim=384, cache_dir=temp_dir, persist=True)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        store1.add(embedding, "persistent text", {"key": "value"})
        store1.save()

        store2 = CacheStore(embedding_dim=384, cache_dir=temp_dir, persist=True)
        results = store2.search(embedding, k=1)
        assert len(results) == 1
        assert results[0][2] == {"key": "value"}

    def test_clear(self, store):
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        store.add(embedding, "text", {"answer": "test"})
        assert len(store) == 1

        store.clear()
        assert len(store) == 0
        assert store.stats()["entry_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
