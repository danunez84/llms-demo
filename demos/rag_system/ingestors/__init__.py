"""Ingestors package.

Exports the base interface and all built-in ingestor implementations.
Students can import BaseIngestor to create new source types.
"""

from .base import BaseIngestor
from .wikipedia import WikipediaIngestor
from .directory import DirectoryIngestor

__all__ = ["BaseIngestor", "WikipediaIngestor", "DirectoryIngestor"]
