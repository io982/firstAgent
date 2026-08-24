"""
Компоненты Главы 3: управление контекстом, памятью и безопасностью.
"""
from .context import (
    Conversation,
    estimate_messages_tokens,
    estimate_tokens,
    smart_trim_history,
    summarize_history,
    trim_by_tokens,
    trim_history,
)
from .memory import LongTermMemory, get_memory
from .security import (
    CONTEXT_RULES,
    INJECTION_PATTERNS,
    SECURITY_RULES,
    looks_like_instruction,
    sanitize_summary,
    sanitize_tool_output,
)

__all__ = [
    "estimate_tokens",
    "estimate_messages_tokens",
    "trim_history",
    "trim_by_tokens",
    "smart_trim_history",
    "Conversation",
    "summarize_history",
    "sanitize_tool_output",
    "sanitize_summary",
    "looks_like_instruction",
    "INJECTION_PATTERNS",
    "CONTEXT_RULES",
    "SECURITY_RULES",
    "LongTermMemory",
    "get_memory",
]
