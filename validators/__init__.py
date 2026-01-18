"""
Validators package: Quality control for the graphic novel pipeline.

Re-exports all validators for convenient importing:
    from validators import PanelValidator, ConsistencyAuditor, ContinuityValidator, ImageCompositionAnalyzer
"""

# Pre-validation (before LLM calls)
from .pre_validators import (
    PreValidationResult,
    PromptPreValidator,
    create_pre_validator_from_assets,
    CharacterState,
    ContinuityValidator,
    validate_script_continuity,
)

# Post-validation (after LLM generation)
from .post_validators import (
    PanelValidationResult,
    PanelValidator,
    ValidationIssue,
    ValidationReport,
    ScriptValidator,
    validate_and_autofix_script,
)

# Image composition analysis
from .composition import (
    CompositionAnalysisResult,
    ImageCompositionAnalyzer,
)

# Consistency auditing
from .consistency import (
    ConsistencyAuditResult,
    ConsistencyAuditor,
)

__all__ = [
    # Pre-validation
    "PreValidationResult",
    "PromptPreValidator",
    "create_pre_validator_from_assets",
    "CharacterState",
    "ContinuityValidator",
    "validate_script_continuity",
    # Post-validation
    "PanelValidationResult",
    "PanelValidator",
    "ValidationIssue",
    "ValidationReport",
    "ScriptValidator",
    "validate_and_autofix_script",
    # Composition
    "CompositionAnalysisResult",
    "ImageCompositionAnalyzer",
    # Consistency
    "ConsistencyAuditResult",
    "ConsistencyAuditor",
]
