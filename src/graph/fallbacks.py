"""Answers the service can still give when no language model is available."""

from src.llm.langchain_llm import _append_sources

EXTRACTIVE_NOTE = (
    "The language model is temporarily unavailable. "
    "Here is the most relevant retrieved evidence."
)
_MAX_EXTRACTIVE_CHUNKS = 3
_MAX_EXCERPT_CHARS = 600


def extractive_answer(chunks: list[dict]) -> str:
    """Quote the top retrieved chunks, labeled so their citations resolve.

    Args:
        chunks: The retrieved chunks, best first. Must not be empty.

    Returns:
        The note and excerpts, followed by the usual sources footer. The quotes
        are not normalized, because that could rewrite what the source says.
    """
    excerpts = [
        f"[SOURCE {index}] {chunk.get('text', '')[:_MAX_EXCERPT_CHARS].strip()}"
        for index, chunk in enumerate(chunks[:_MAX_EXTRACTIVE_CHUNKS], start=1)
    ]
    return _append_sources("\n\n".join([EXTRACTIVE_NOTE, *excerpts]), chunks)
