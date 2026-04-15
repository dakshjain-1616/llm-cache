"""
Example demonstrating OpenAI wrapper with semantic caching.

This example shows how to use the CachedOpenAI wrapper to automatically
cache responses and demonstrate cache hits/misses.

Note: This example uses mock responses for demonstration without requiring
actual API keys. In production, you would use real API calls.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_cache import CachedOpenAI, SemanticCache


class MockOpenAIResponse:
    """Mock OpenAI response for demonstration."""
    
    def __init__(self, content: str, model: str = "gpt-4"):
        self.choices = [MockChoice(content)]
        self.model = model
        self.object = "chat.completion"
    
    def __repr__(self):
        return f"MockOpenAIResponse(content='{self.choices[0].message.content[:50]}...')"


class MockChoice:
    """Mock choice object."""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)
        self.finish_reason = "stop"
        self.index = 0


class MockMessage:
    """Mock message object."""
    
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"


def mock_api_call(prompt: str, model: str = "gpt-4") -> MockOpenAIResponse:
    """Simulate an API call with a mock response."""
    responses = {
        "capital of France": "The capital of France is Paris.",
        "capital of Germany": "The capital of Germany is Berlin.",
        "largest planet": "The largest planet in our solar system is Jupiter.",
        "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    }
    
    # Find matching response or generate generic one
    content = "I'm not sure about that."
    for key, value in responses.items():
        if key.lower() in prompt.lower():
            content = value
            break
    
    return MockOpenAIResponse(content, model)


def demonstrate_cache():
    """Demonstrate cache hits and misses."""
    print("=" * 60)
    print("OpenAI Semantic Cache Demo")
    print("=" * 60)
    
    # Create a cache with a lower threshold for demo purposes
    # (In production, you'd use a higher threshold like 0.95)
    cache = SemanticCache(threshold=0.90, persist=False)
    
    # Create a mock client that wraps our cache
    class DemoCachedOpenAI(CachedOpenAI):
        """Demo client that uses mock responses."""
        
        def __init__(self, cache):
            # Don't call super().__init__ to avoid needing real API key
            self._cache = cache
            self._client = None
        
        @property
        def chat(self):
            return DemoCachedChatCompletions(self._cache)
    
    class DemoCachedChatCompletions:
        """Demo chat completions with mock API."""
        
        def __init__(self, cache):
            self._cache = cache
        
        def create(self, model: str, messages: list, **kwargs):
            # Check cache first
            cached = self._cache.get(messages, model=model, **kwargs)
            if cached is not None:
                print(f"  [CACHE HIT] Similarity-based match found!")
                return cached
            
            # Simulate API call
            prompt_text = messages[0]["content"] if messages else ""
            print(f"  [CACHE MISS] Calling mock API...")
            response = mock_api_call(prompt_text, model)
            
            # Cache the response
            self._cache.set(messages, response, model=model, **kwargs)
            return response
    
    client = DemoCachedOpenAI(cache)
    
    # Test prompts - similar but not identical
    prompts = [
        "What is the capital of France?",
        "Tell me the capital of France.",  # Should hit cache (semantically similar)
        "What is the capital of Germany?",
        "What's the capital city of France?",  # Should hit cache
        "Which planet is the largest in our solar system?",
        "What is the speed of light?",
        "How fast does light travel?",  # Should hit cache
    ]
    
    print("\nSending prompts (with semantic caching enabled):\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. Prompt: '{prompt}'")
        
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.create(
            model="gpt-4",
            messages=messages
        )
        
        print(f"   Response: {response.choices[0].message.content}")
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
    query = "France capital"
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
