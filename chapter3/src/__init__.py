"""
Компоненты Главы 3: управление контекстом, памятью и безопасностью.
"""
from .context import (
    OBSERVATION_PREFIX,
    Conversation,
    drop_orphan_observations,
    estimate_messages_tokens,
    estimate_tokens,
    is_observation,
    smart_trim_history,
    summarize_history,
    trim_by_tokens,
    trim_history,
)
from .core_memory import (
    BLOCK_LIMIT,
    CORE_FIELDS,
    FIELD_LIMIT,
    CoreMemory,
    get_core_memory,
)
from .memory import LongTermMemory, get_memory
from .security import (
    CONTEXT_RULES,
    INJECTION_PATTERNS,
    SECURITY_RULES,
    looks_like_instruction,
    sanitize_core_memory,
    sanitize_summary,
    sanitize_tool_output,
)

__all__ = [
    "estimate_tokens",
    "estimate_messages_tokens",
    "trim_history",
    "trim_by_tokens",
    "smart_trim_history",
    "drop_orphan_observations",
    "is_observation",
    "OBSERVATION_PREFIX",
    "Conversation",
    "summarize_history",
    "sanitize_tool_output",
    "sanitize_summary",
    "sanitize_core_memory",
    "looks_like_instruction",
    "INJECTION_PATTERNS",
    "CONTEXT_RULES",
    "SECURITY_RULES",
    "LongTermMemory",
    "get_memory",
    "CoreMemory",
    "get_core_memory",
    "CORE_FIELDS",
    "FIELD_LIMIT",
    "BLOCK_LIMIT",
]
