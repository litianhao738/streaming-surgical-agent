"""Opt-in local knowledge retrieval; no changes to the frozen repair pipeline."""

from .graph_review import RetrievalPolicy, knowledge_manifest, retrieve

__all__ = ["RetrievalPolicy", "knowledge_manifest", "retrieve"]
