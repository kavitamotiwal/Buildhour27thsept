class RagChatError(Exception):
    """Base for every error this package raises on purpose."""


class ConfigurationError(RagChatError):
    """A required key, model, or backend is missing. Retrying will not help."""


class IngestError(RagChatError):
    """A document could not be loaded or chunked. Names the offending file."""


class ProviderError(RagChatError):
    """An embedding or LLM provider failed after retries were exhausted."""


class IndexMissingError(RagChatError):
    """No vector index is present. Message tells the user how to build one."""
