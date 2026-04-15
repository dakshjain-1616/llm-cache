"""Tests for Anthropic wrapper classes."""

import asyncio
import pytest
from unittest.mock import Mock

from llm_cache.wrappers.anthropic_wrapper import (
    CachedAnthropic,
    CachedMessages,
    AsyncCachedAnthropic,
    AsyncCachedMessages,
)


class MockContentBlock:
    def __init__(self, text="Test response"):
        self.type = "text"
        self.text = text


class MockUsage:
    def __init__(self):
        self.input_tokens = 10
        self.output_tokens = 5


class MockAnthropicResponse:
    def __init__(self, text="Test response"):
        self.id = "msg_test123"
        self.type = "message"
        self.role = "assistant"
        self.content = [MockContentBlock(text)]
        self.model = "claude-3-haiku-20240307"
        self.stop_reason = "end_turn"
        self.usage = MockUsage()


class TestCachedAnthropic:
    """Test CachedAnthropic wrapper."""

    def test_init(self):
        """Test initialization with threshold kwarg."""
        mock_client = Mock()
        wrapper = CachedAnthropic(
            client=mock_client,
            threshold=0.9,
        )

        assert wrapper._client == mock_client
        assert wrapper._cache is not None
        assert wrapper._cache.threshold == 0.9

    def test_messages_property(self):
        """Test .messages returns CachedMessages."""
        mock_client = Mock()
        wrapper = CachedAnthropic(client=mock_client)

        assert isinstance(wrapper.messages, CachedMessages)

    def test_get_stats(self):
        """Test get_stats returns expected keys."""
        mock_client = Mock()
        wrapper = CachedAnthropic(client=mock_client)

        stats = wrapper.get_stats()
        assert "hits" in stats
        assert "misses" in stats

    def test_clear_cache(self):
        """Test clear_cache does not raise."""
        mock_client = Mock()
        wrapper = CachedAnthropic(client=mock_client)
        wrapper.clear_cache()


class TestCachedMessages:
    """Test CachedMessages via client.messages.create()."""

    def test_create_cache_miss(self):
        """First call should hit the real API (cache miss)."""
        mock_client = Mock()
        mock_response = MockAnthropicResponse("Test response")
        mock_client.messages.create.return_value = mock_response

        wrapper = CachedAnthropic(client=mock_client)

        response = wrapper.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response == mock_response
        mock_client.messages.create.assert_called_once()

    def test_create_cache_hit(self):
        """Second identical call should return cached result without hitting API."""
        mock_client = Mock()
        mock_response = MockAnthropicResponse("Cached response")
        mock_client.messages.create.return_value = mock_response

        wrapper = CachedAnthropic(client=mock_client)

        # First call — cache miss
        response1 = wrapper.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Second call with identical prompt — should be a cache hit
        response2 = wrapper.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response1 == response2
        # API called only once
        mock_client.messages.create.assert_called_once()

    def test_create_streaming_bypasses_cache(self):
        """Streaming requests should always bypass the cache."""
        mock_client = Mock()
        mock_response = MockAnthropicResponse("Streaming response")
        mock_client.messages.create.return_value = mock_response

        wrapper = CachedAnthropic(client=mock_client)

        response = wrapper.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        assert response == mock_response
        mock_client.messages.create.assert_called_once()

    def test_create_with_system_prompt(self):
        """create() with a system prompt should work without error."""
        mock_client = Mock()
        mock_response = MockAnthropicResponse("System response")
        mock_client.messages.create.return_value = mock_response

        wrapper = CachedAnthropic(client=mock_client)

        response = wrapper.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
        )

        assert response == mock_response


class TestAsyncCachedAnthropic:
    """Test AsyncCachedAnthropic wrapper (sync entry points using asyncio.run)."""

    def test_init(self):
        """Test initialization with threshold kwarg."""
        mock_client = Mock()
        wrapper = AsyncCachedAnthropic(
            client=mock_client,
            threshold=0.9,
        )

        assert wrapper._client == mock_client
        assert wrapper._cache is not None
        assert wrapper._cache.threshold == 0.9

    def test_messages_property(self):
        """Test .messages returns AsyncCachedMessages."""
        mock_client = Mock()
        wrapper = AsyncCachedAnthropic(client=mock_client)

        assert isinstance(wrapper.messages, AsyncCachedMessages)

    def test_create_cache_miss(self):
        """Async create with cache miss should call the underlying client."""
        mock_client = Mock()
        mock_response = MockAnthropicResponse("Async response")

        async def async_create(*args, **kwargs):
            return mock_response

        mock_client.messages.create = async_create

        wrapper = AsyncCachedAnthropic(client=mock_client)

        async def run():
            return await wrapper.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello async"}],
            )

        response = asyncio.run(run())
        assert response == mock_response
