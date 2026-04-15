"""Utility functions for LLM Cache."""

import hashlib
import json
import pickle
from typing import Any, Dict, List, Optional, Union


def hash_prompt(prompt: str, model: Optional[str] = None) -> str:
    """
    Create a deterministic hash for a prompt.
    
    Args:
        prompt: The prompt text to hash
        model: Optional model name to include in hash
        
    Returns:
        Hex digest of the hash
    """
    content = prompt
    if model:
        content = f"{model}:{prompt}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def serialize_response(response: Any) -> bytes:
    """
    Serialize a response object to bytes using pickle.
    
    Args:
        response: The response object to serialize
        
    Returns:
        Pickled bytes
    """
    return pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_response(data: bytes) -> Any:
    """
    Deserialize response bytes back to object.
    
    Args:
        data: Pickled bytes
        
    Returns:
        Deserialized object
    """
    return pickle.loads(data)


def format_prompt(
    messages: Union[str, List[Dict[str, str]], Dict[str, Any]],
    system: Optional[str] = None
) -> str:
    """
    Normalize various prompt formats into a canonical string representation.
    
    Supports:
    - Simple string prompts
    - OpenAI-style message lists: [{"role": "user", "content": "..."}]
    - Anthropic-style with system parameter
    
    Args:
        messages: The prompt in various formats
        system: Optional system message (for Anthropic-style)
        
    Returns:
        Normalized string representation
    """
    if isinstance(messages, str):
        if system:
            return f"System: {system}\n\nUser: {messages}"
        return messages
    
    if isinstance(messages, list):
        parts = []
        if system:
            parts.append(f"System: {system}")
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                parts.append(f"{role.capitalize()}: {content}")
            else:
                parts.append(str(msg))
        return "\n\n".join(parts)
    
    if isinstance(messages, dict):
        # Handle single message dict
        if 'content' in messages:
            role = messages.get('role', 'user')
            content = messages['content']
            if system:
                return f"System: {system}\n\n{role.capitalize()}: {content}"
            return f"{role.capitalize()}: {content}"
        # Handle dict with messages key
        if 'messages' in messages:
            return format_prompt(messages['messages'], system or messages.get('system'))
    
    return str(messages)


def extract_response_text(response: Any) -> str:
    """
    Extract text content from various response formats.
    
    Args:
        response: Response object from OpenAI or Anthropic
        
    Returns:
        Extracted text content
    """
    # Handle OpenAI response format
    if hasattr(response, 'choices') and response.choices:
        choice = response.choices[0]
        if hasattr(choice, 'message') and choice.message:
            return choice.message.content or ""
        if hasattr(choice, 'text'):
            return choice.text or ""
    
    # Handle Anthropic response format
    if hasattr(response, 'content') and response.content:
        if isinstance(response.content, list):
            texts = []
            for item in response.content:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                elif isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
            return "\n".join(texts)
        return str(response.content)
    
    # Fallback: try to convert to string
    return str(response)


def cosine_similarity(a: Any, b: Any) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector (numpy array or list)
        b: Second vector (numpy array or list)
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    import numpy as np
    
    a = np.array(a)
    b = np.array(b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use as a filename.
    
    Args:
        name: The string to sanitize
        
    Returns:
        Sanitized string
    """
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:100]  # Limit length
