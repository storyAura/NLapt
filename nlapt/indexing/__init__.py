"""Search and sorting over dataset keys (contract: docs/ARCHITECTURE.md,
sections ``nlapt.indexing.search_index`` and ``nlapt.indexing.sorting``).
"""

from __future__ import annotations

from nlapt.indexing.search_index import SearchIndex
from nlapt.indexing.sorting import SortBy, natural_key, sort_keys

__all__ = [
    "SearchIndex",
    "SortBy",
    "natural_key",
    "sort_keys",
]
