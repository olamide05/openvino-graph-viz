"""Graph transformation passes, the Pipeline that runs them in order, and the pass registry.
"""
from .base import Pass, Pipeline, register_pass, get_pass, available_passes
from .highlight import HighlightPass
from .hide_constants import HideConstantsPass
from .fold_constant_converts import FoldConstantConvertsPass
from .collapse_fanout import CollapseFanoutPass
from .collapse_modules import CollapseModulesPass

__all__ = ["Pass", "Pipeline", "register_pass", "get_pass", "available_passes",
           "HighlightPass", "HideConstantsPass", "FoldConstantConvertsPass",
           "CollapseFanoutPass", "CollapseModulesPass"]
