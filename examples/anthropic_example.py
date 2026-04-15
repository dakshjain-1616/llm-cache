"""
Example demonstrating Anthropic wrapper with semantic caching.

This example shows how to use the CachedAnthropic wrapper to automatically
cache responses and demonstrate cache hits/misses.

Note: This example uses mock responses for demonstration without requiring
actual API keys. In production, you would use real API calls.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_cache import CachedAnthropic, SemanticCache


class MockAnthropicResponse:
    """Mock Anthropic response for demonstration."""
    
    def __init__(self, content: str, model: str = "claude-3-opus-20240229"):
        self.content = [MockContentBlock(content)]
        self.model = model
        self.id = f"mock-{hash(content) & 0xFFFFFFFF:08x}"
        self.type = "message"
        self.role = "assistant"
        self.stop_reason = "end_turn"
    
    def __repr__(self):
        return f"MockAnthropicResponse(content='{self.content[0].text[:50]}...')"


class MockContentBlock:
    """Mock content block."""
    
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


def mock_api_call(prompt: str, model: str = "claude-3-opus-20240229") -> MockAnthropicResponse:
    """Simulate an API call with a mock response."""
    responses = {
        "python list comprehension": "List comprehensions in Python provide a concise way to create lists. Example: `[x**2 for x in range(10)]` creates squares of 0-9.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming. It includes supervised, unsupervised, and reinforcement learning.",
        "neural network": "Neural networks are computing systems inspired by biological neurons. They consist of layers of interconnected nodes that process information.",
        "deep learning": "Deep learning uses neural networks with many layers (hence 'deep') to model complex patterns. It's particularly effective for image recognition, NLP, and more.",
    }
    
    # Find matching response or generate generic one
    content = "I don't have specific information about that topic."
    for key, value in responses.items():
        if key.lower() in prompt.lower():
            content = value
            break
    
    return MockAnthropicResponse(content, model)


def demonstrate_cache():
    """Demonstrate cache hits and misses."""
    print("=" * 60)
    print("Anthropic Semantic Cache Demo")
    print("=" * 60)
    
    # Create a cache with a lower threshold for demo purposes
    cache = SemanticCache(threshold=0.90, persist=False)
    
    # Create a mock client that wraps our cache
    class DemoCachedAnthropic(CachedAnthropic):
        """Demo client that uses mock responses."""
        
        def __init__(self, cache):
            # Don't call super().__init__ to avoid needing real API key
            self._cache = cache
            self._client = None
        
        @property
        def messages(self):
            return DemoCachedMessages(self._cache)
    
    class DemoCachedMessages:
        """Demo messages with mock API."""
        
        def __init__(self, cache):
            self._cache = cache
        
        def create(self, model: str, max_tokens: int, messages: list, system: str = None, **kwargs):
            # Check cache first
            cached = self._cache.get(messages, system=system, model=model, **kwargs)
            if cached is not None:
                print(f"  [CACHE HIT] Similarity-based match found!")
                return cached
            
            # Simulate API call
            prompt_text = messages[0]["content"] if messages else ""
            print(f"  [CACHE MISS] Calling mock API...")
            response = mock_api_call(prompt_text, model)
            
            # Cache the response
            self._cache.set(messages, response, system=system, model=model, **kwargs)
            return response
    
    client = DemoCachedAnthropic(cache)
    
    # Test prompts - similar but not identical
    prompts = [
        ("Explain Python list comprehensions", None),
        ("How do list comprehensions work in Python?", None),  # Should hit cache
        ("What is machine learning?", None),
        ("Tell me about ML and AI", None),  # Might hit cache (similar to machine learning)
        ("Explain neural networks", "You are a helpful AI assistant."),
        ("What are deep neural networks?", "You are a helpful AI assistant."),  # Should hit
        ("Describe deep learning", None),  # Should hit cache (similar to neural networks)
    ]
    
    print("\nSending prompts (with semantic caching enabled):\n")
    
    for i, (prompt, system) in enumerate(prompts, 1):
        system_str = f" [System: '{system[:30]}...']" if system else ""
        print(f"{i}. Prompt: '{prompt}'{system_str}")
        
        messages = [{"role": "user", "content": prompt}]
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=messages,
            system=system
        )
        
        print(f"   Response: {response.content[0].text}")
        print()
    
    # Print cache statistics
    print("-" * 60)
    print("Cache Statistics:")
    stats = cache.stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Similarity threshold: {stats['threshold']}")
    print("-" * 60)
    
    # Demonstrate similarity search
    print("\nSimilarity Search Demo:")
    query = "Python programming"
    print(f"Query: '{query}'")
    similar = cache.get_similar(query, k=3)
    print("Most similar cached entries:")
    for entry in similar:
        print(f"  - Similarity: {entry['similarity']:.4f}")
        print(f"    Text: {entry['text'][:60]}...")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_cache()
