"""Right-panel tool section widgets (查找替换 / 前缀后缀 / 翻译对照 / 历史记录)."""

from __future__ import annotations

from nlapt_gui.widgets.sections.find_replace import FindReplaceSection
from nlapt_gui.widgets.sections.history import HistorySection
from nlapt_gui.widgets.sections.prefix_suffix import PrefixSuffixSection
from nlapt_gui.widgets.sections.translate import TranslateSection

__all__ = [
    "FindReplaceSection",
    "HistorySection",
    "PrefixSuffixSection",
    "TranslateSection",
]
