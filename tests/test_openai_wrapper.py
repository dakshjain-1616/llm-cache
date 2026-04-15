"""Tests for OpenAI wrapper classes."""

import asyncio
import pytest
from unittest.mock import Mock

from llm_cache.wrappers.openai_wrapper import (
    CachedOpenAI,
    CachedChat,
    CachedChatCompletions,
    CachedCompletions,
    AsyncCachedOpenAI,
    AsyncCachedChat,
    AsyncCachedChatCompletions,
    AsyncCachedCompletions,
)


class MockMessage:
    def __init__(self, content="Test response"):
        self.content = content
        self.role = "assistant"


class MockUsage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


class MockChoice:
    def __init__(self, content="Test response"):
        self.message = MockMessage(content)
        self.finish_reason = "stop"


class MockOpenAIResponse:
    def __init__(self, content="Test response"):
        self.choices = [MockChoice(content)]
        self.model = "gpt-3.5-turbo"
        self.usage = MockUsage()


class TestCachedOpenAI:
    """Test CachedOpenAI wrapper."""

    def test_init(self):
        """Test initialization with threshold and cache_dir kwargs."""
        mock_client = Mock()
        wrapper = CachedOpenAI(
            client=mock_client,
            threshold=0.9,
            cache_dir="/tmp/test_llm_cache_openai",
        )

        assert wrapper._client == mock_client
        assert wrapper._cache is not None
        assert wrapper._cache.threshold == 0.9

    def test_chat_property_returns_cached_chat(self):
        """Test that .chat returns a CachedChat (not CachedChatCompletions directly)."""
        mock_client = Mock()
        wrapper = CachedOpenAI(client=mock_client)

        assert isinstance(wrapper.chat, CachedChat)

    def test_chat_completions_property(self):
        """Test that client.chat.completions returns CachedChatCompletions."""
        mock_client = Mock()
        wrapper = CachedOpenAI(client=mock_client)

        chat_completions = wrapper.chat.completions
        assert isinstance(chat_completions, CachedChatCompletions)

    def test_completions_property(self):
        """Test .completions returns CachedCompletions."""
        mock_client = Mock()
        wrapper = CachedOpenAI(client=mock_client)

        completions = wrapper.completions
        assert isinstance(completions, CachedCompletions)

    def test_get_stats(self):
        """Test get_stats method returns expected keys."""
        mock_client = Mock()
        wrapper = CachedOpenAI(client=mock_client)

        stats = wrapper.get_stats()
        assert "hits" in stats
        assert "misses" in stats

    def test_clear_cache(self):
        """Test clear_cache does not raise."""
        mock_client = Mock()
        wrapper = CachedOpenAI(client=mock_client)
        wrapper.clear_cache()


class TestCachedChatCompletions:
    """Test CachedChatCompletions via client.chat.completions.create()."""

    def test_create_cache_miss(self):
        """First call should hit the real API (cache miss)."""
        mock_client = Mock()
        mock_response = MockOpenAIResponse("Test response")
        mock_client.chat.completions.create.return_value = mock_response

        wrapper = CachedOpenAI(client=mock_client)

        response = wrapper.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response == mock_response
        mock_client.chat.completions.create.assert_called_once()

    def test_create_cache_hit(self):
        """Second identical call should return cached result without hitting API."""
        mock_client = Mock()
        mock_response = MockOpenAIResponse("Cached response")
        mock_client.chat.completions.create.return_value = mock_response

        wrapper = CachedOpenAI(client=mock_client)

        # First call — cache miss
        response1 = wrapper.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Second call with identical prompt — cache hit
        response2 = wrapper.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response1 == response2
        # API called only once
        mock_client.chat.completions.create.assert_called_once()

    def test_create_streaming_bypasses_cache(self):
        """Streaming requests should always bypass the cache."""
        mock_client = Mock()
        mock_response = MockOpenAIResponse("Streaming response")
        mock_client.chat.completions.create.return_value = mock_response

        wrapper = CachedOpenAI(client=mock_client)

        response = wrapper.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        assert response == mock_response
        mock_client.chat.completions.create.assert_called_once()

    def test_create_with_system_prompt(self):
        """System message in the messages list should work without error."""
        mock_client = Mock()
        mock_response = MockOpenAIResponse("Test response")
        mock_client.chat.completions.create.return_value = mock_response

        wrapper = CachedOpenAI(client=mock_client)

        response = wrapper.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
        )

        assert response == mock_response


class TestCachedCompletions:
    """Test legacy CachedCompletions."""

    def test_create_cache_miss(self):
        """First call should hit the real API."""
        mock_client = Mock()
        mock_response = MockOpenAIResponse("Test completion")
        mock_client.completions.create.return_value = mock_response

        wrapper = CachedOpenAI(client=mock_client)

        response = wrapper.completions.create(
            model="text-davinci-003",
            prompt="Hello world",
        )

        assert response == mock_response
        mock_client.completions.create.assert_called_once()


class TestAsyncCachedOpenAI:
    """Test AsyncCachedOpenAI wrapper (sync entry points using asyncio.run)."""

    def test_init(self):
        """Test initialization with threshold kwarg."""
        mock_client = Mock()
        wrapper = AsyncCachedOpenAI(
            client=mock_client,
            threshold=0.9,
        )

        assert wrapper._client == mock_client
        assert wrapper._cache is not None
        assert wrapper._cache.threshold == 0.9

    def test_chat_property_returns_async_cached_chat(self):
        """Test .chat returns AsyncCachedChat."""
        mock_client = Mock()
        wrapper = AsyncCachedOpenAI(client=mock_client)

        assert isinstance(wrapper.chat, AsyncCachedChat)

    def test_chat_completions_property(self):
        """Test .chat.completions returns AsyncCachedChatCompletions."""
        mock_client = Mock()
        wrapper = AsyncCachedOpenAI(client=mock_client)

        chat_completions = wrapper.chat.completions
        assert isinstance(chat_completions, AsyncCachedChatCompletions)

    def test_create_cache_miss(self):
        """Async create with cache miss should call the underlying client."""
        mock_client = Mock()
        mock_response = MockOpenAIResponse("Async response")

        async def async_create(*args, **kwargs):
            return mock_response

        mock_client.chat.completions.create = async_create

        wrapper = AsyncCachedOpenAI(client=mock_client)

        async def run():
            return await wrapper.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello async"}],
            )

        response = asyncio.run(run())
        assert response == mock_response


class TestAsyncCachedCompletions:
    """Test AsyncCachedCompletions."""

    def test_create_cache_miss(self):
        """Async legacy completion with cache miss should call the underlying client."""
        mock_client = Mock()
        mock_response = MockOpenAIResponse("Async completion")

        async def async_create(*args, **kwargs):
            return mock_response

        mock_client.completions.create = async_create

        wrapper = AsyncCachedOpenAI(client=mock_client)

        async def run():
            return await wrapper.completions.create(
                model="text-davinci-003",
                prompt="Hello async",
            )

        response = asyncio.run(run())
        assert response == mock_response
