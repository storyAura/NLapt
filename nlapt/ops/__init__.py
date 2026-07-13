"""Pure text operations: find/replace, prefix/suffix, tag dedup.

Re-exports the public API of the ``nlapt.ops`` package (contract:
docs/ARCHITECTURE.md, sections ``nlapt.ops.*``).
"""

from __future__ import annotations

from nlapt.ops.base import (
    CONTEXT_CHARS,
    OP_DEDUP_TAGS,
    OP_FIND_REPLACE,
    OP_PREFIX_SUFFIX,
    MatchPreview,
    OperationRegistry,
    TextOperation,
    get_default_registry,
)
from nlapt.ops.dedup import DedupTagsOperation
from nlapt.ops.find_replace import (
    FindReplaceOperation,
    FindReplaceSpec,
    ScopeHitStats,
    compile_spec,
    scope_stats,
)
from nlapt.ops.prefix_suffix import (
    JOINERS,
    PrefixSuffixOperation,
    PrefixSuffixSpec,
    make_trigger_add_op,
    make_trigger_remove_op,
    remove_exact_prefix,
    remove_exact_suffix,
)

__all__ = [
    "CONTEXT_CHARS",
    "OP_DEDUP_TAGS",
    "OP_FIND_REPLACE",
    "OP_PREFIX_SUFFIX",
    "MatchPreview",
    "OperationRegistry",
    "TextOperation",
    "get_default_registry",
    "DedupTagsOperation",
    "FindReplaceOperation",
    "FindReplaceSpec",
    "ScopeHitStats",
    "compile_spec",
    "scope_stats",
    "JOINERS",
    "PrefixSuffixOperation",
    "PrefixSuffixSpec",
    "make_trigger_add_op",
    "make_trigger_remove_op",
    "remove_exact_prefix",
    "remove_exact_suffix",
]
