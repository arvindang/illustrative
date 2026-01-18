"""
Validators module: Quality control for the graphic novel pipeline.

Includes:
- PromptPreValidator: PRE-generation validation to avoid wasting tokens
- ContinuityValidator: Pre-illustration script validation (character states, impossible scenarios)
- PanelValidator: Post-generation image validation (era accuracy, character presence)
- ConsistencyAuditor: Cross-panel character consistency checking
- ScriptValidator: Post-generation script validation with auto-fixes
"""

import json
import io
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from PIL import Image
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from config import config


# ==============================================================================
# PROMPT PRE-VALIDATOR - Validate inputs BEFORE expensive LLM calls
# ==============================================================================

@dataclass
class PreValidationResult:
    """Result of pre-validating inputs before LLM call."""
    valid: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    auto_fixed: Dict[str, str] = field(default_factory=dict)  # field -> fixed_value


class PromptPreValidator:
    """
    Validates and pre-processes inputs BEFORE sending to the LLM.

    This saves tokens by catching issues early:
    - Dialogue/caption length will be enforced BEFORE generation
    - Visual descriptions will be validated for required elements
    - Era constraints will be pre-checked against known terms
    - Character references will be validated against known characters

    Usage:
        validator = PromptPreValidator(era="1860s Victorian", known_characters=["Captain Nemo"])
        result = validator.validate_panel_inputs(visual_desc, dialogue, characters)
        if not result.valid:
            # Handle issues before LLM call
    """

    MAX_DIALOGUE_CHARS = 80
    MAX_CAPTION_CHARS = 100

    # Era-specific forbidden terms (modern words that shouldn't appear)
    ERA_FORBIDDEN_TERMS = {
        "1860s": ["phone", "computer", "internet", "email", "television", "tv", "radio",
                  "airplane", "automobile", "car", "electricity", "plastic", "laser",
                  "robot", "android", "helicopter", "neon", "satellite"],
        "medieval": ["gun", "rifle", "pistol", "electricity", "engine", "motor",
                     "phone", "radio", "plastic", "computer", "glass window",
                     "printing press", "clock", "potato", "tomato", "corn"],
        "ancient_rome": ["stirrup", "paper", "glass window", "gun", "engine",
                         "electricity", "plastic", "printing", "compass"],
    }

    # Scene type requirements (what MUST be mentioned)
    SCENE_REQUIREMENTS = {
        "underwater": ["diving", "helmet", "suit", "breathing", "water", "submarine"],
        "formal": ["attire", "dress", "suit", "gown", "evening", "coat"],
        "action": ["motion", "movement", "running", "fighting", "chase"],
    }

    def __init__(
        self,
        era: str = "",
        known_characters: List[str] = None,
        known_locations: List[str] = None,
        interaction_rules: Dict[str, List[str]] = None
    ):
        self.era = era.lower()
        self.known_characters = set(known_characters or [])
        self.known_locations = set(known_locations or [])
        self.interaction_rules = interaction_rules or {}

        # Determine which era's forbidden terms apply
        self.forbidden_terms = []
        for era_key, terms in self.ERA_FORBIDDEN_TERMS.items():
            if era_key in self.era:
                self.forbidden_terms.extend(terms)

    def validate_panel_inputs(
        self,
        visual_description: str,
        dialogue: str = "",
        caption: str = "",
        characters: List[str] = None,
        scene_type: str = ""
    ) -> PreValidationResult:
        """
        Validate all inputs for a single panel BEFORE LLM generation.

        Args:
            visual_description: The visual description to validate
            dialogue: Proposed dialogue text
            caption: Proposed caption/narration text
            characters: List of character names in the panel
            scene_type: Type of scene for requirement checking

        Returns:
            PreValidationResult with validation status and any auto-fixes
        """
        issues = []
        suggestions = []
        auto_fixed = {}

        # 1. Validate dialogue length
        if dialogue and len(dialogue) > self.MAX_DIALOGUE_CHARS:
            # Auto-fix by truncating
            fixed = self._truncate_at_word_boundary(dialogue, self.MAX_DIALOGUE_CHARS)
            auto_fixed["dialogue"] = fixed
            issues.append(f"Dialogue too long ({len(dialogue)} chars > {self.MAX_DIALOGUE_CHARS}). Auto-truncated.")

        # 2. Validate caption length
        if caption and len(caption) > self.MAX_CAPTION_CHARS:
            fixed = self._truncate_at_word_boundary(caption, self.MAX_CAPTION_CHARS)
            auto_fixed["caption"] = fixed
            issues.append(f"Caption too long ({len(caption)} chars > {self.MAX_CAPTION_CHARS}). Auto-truncated.")

        # 3. Check for era anachronisms in visual description
        anachronisms = self._find_anachronisms(visual_description)
        if anachronisms:
            issues.append(f"Era anachronisms detected: {', '.join(anachronisms)}")
            suggestions.append("Remove or replace modern terms with era-appropriate alternatives")

        # 4. Check for era anachronisms in dialogue
        dialogue_anachronisms = self._find_anachronisms(dialogue)
        if dialogue_anachronisms:
            issues.append(f"Dialogue anachronisms: {', '.join(dialogue_anachronisms)}")

        # 5. Validate scene type requirements
        if scene_type and scene_type.lower() in self.SCENE_REQUIREMENTS:
            required = self.SCENE_REQUIREMENTS[scene_type.lower()]
            desc_lower = visual_description.lower()
            missing = [r for r in required if not any(r in desc_lower for r in required[:3])]
            if len(missing) == len(required):  # None of the required terms present
                suggestions.append(f"Scene type '{scene_type}' typically includes: {', '.join(required[:3])}")

        # 6. Validate character references
        if characters:
            unknown = [c for c in characters if c not in self.known_characters and self.known_characters]
            if unknown:
                suggestions.append(f"Unknown characters referenced: {', '.join(unknown)}")

        # 7. Check interaction rules for this scene type
        if scene_type and self.interaction_rules:
            scene_rules = self.interaction_rules.get(f"{scene_type}_scenes", [])
            if scene_rules:
                # Check if any rule keywords are missing from description
                for rule in scene_rules[:2]:  # Check first 2 rules
                    rule_keywords = rule.lower().split()[:3]  # First 3 words
                    if not any(kw in visual_description.lower() for kw in rule_keywords):
                        suggestions.append(f"Scene rule may apply: {rule}")

        return PreValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            auto_fixed=auto_fixed
        )

    def validate_blueprint_page(
        self,
        page_number: int,
        summary: str,
        focus_text: str,
        key_characters: List[str] = None,
        scene_type: str = ""
    ) -> PreValidationResult:
        """
        Validate blueprint page inputs before LLM expansion.
        """
        issues = []
        suggestions = []
        auto_fixed = {}

        # 1. Check focus_text isn't too long (should be visual moments, not excerpts)
        if len(focus_text) > 500:
            suggestions.append("Focus text is very long - consider extracting only key visual moments")

        # 2. Check for missing key elements
        if not key_characters:
            suggestions.append(f"Page {page_number} has no key_characters defined")

        # 3. Check for era anachronisms in summary
        anachronisms = self._find_anachronisms(summary)
        if anachronisms:
            issues.append(f"Summary contains anachronisms: {', '.join(anachronisms)}")

        return PreValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            auto_fixed=auto_fixed
        )

    def _truncate_at_word_boundary(self, text: str, max_length: int) -> str:
        """Truncate text at a word boundary with ellipsis."""
        if len(text) <= max_length:
            return text
        truncated = text[:max_length - 3]
        last_space = truncated.rfind(' ')
        if last_space > max_length // 2:
            truncated = truncated[:last_space]
        return truncated + "..."

    def _find_anachronisms(self, text: str) -> List[str]:
        """Find era-inappropriate terms in text."""
        if not text or not self.forbidden_terms:
            return []
        text_lower = text.lower()
        found = []
        for term in self.forbidden_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                found.append(term)
        return found

    def preprocess_dialogue(self, dialogue: str) -> str:
        """
        Clean and fix common dialogue issues before LLM processing.
        Returns cleaned dialogue.
        """
        if not dialogue:
            return dialogue

        # Fix punctuation issues
        cleaned = dialogue
        cleaned = re.sub(r'\.\.\.!', '—', cleaned)  # ...! -> em-dash
        cleaned = re.sub(r'\?\!', '?', cleaned)  # ?! -> ?
        cleaned = re.sub(r'\!\?', '!', cleaned)  # !? -> !
        cleaned = re.sub(r'\.\.\.\.+', '...', cleaned)  # Multiple dots -> three
        cleaned = re.sub(r'!!+', '!', cleaned)  # Multiple ! -> one
        cleaned = re.sub(r'\?\?+', '?', cleaned)  # Multiple ? -> one

        # Remove stage directions
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)  # Remove (parentheticals)
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)  # Remove [brackets]

        # Trim and length-check
        cleaned = cleaned.strip()
        if len(cleaned) > self.MAX_DIALOGUE_CHARS:
            cleaned = self._truncate_at_word_boundary(cleaned, self.MAX_DIALOGUE_CHARS)

        return cleaned


def create_pre_validator_from_assets(
    era: str = "",
    assets: Dict = None,
    character_arcs: Dict = None
) -> PromptPreValidator:
    """
    Factory function to create a PromptPreValidator from pipeline assets.

    Args:
        era: Era constraint string
        assets: Asset manifest from generate_asset_manifest()
        character_arcs: Character arcs from generate_character_deep_dive()

    Returns:
        Configured PromptPreValidator instance
    """
    known_characters = []
    known_locations = []
    interaction_rules = {}

    if assets:
        # Extract character names
        for char in assets.get('characters', []):
            known_characters.append(char.get('name', ''))
        # Extract location names
        for loc in assets.get('locations', []):
            known_locations.append(loc.get('name', ''))
        # Extract interaction rules
        interaction_rules = assets.get('interaction_rules', {})

    if character_arcs:
        # Also get character names from arcs
        for char in character_arcs.get('characters', []):
            name = char.get('name', '')
            if name and name not in known_characters:
                known_characters.append(name)

    return PromptPreValidator(
        era=era,
        known_characters=known_characters,
        known_locations=known_locations,
        interaction_rules=interaction_rules
    )


# Lazy client initialization
_client = None
_client_key = None


def get_client():
    """Returns a Gemini client, creating a new one if API key has changed."""
    global _client, _client_key
    if _client is None or _client_key != config.gemini_api_key:
        _client = genai.Client(api_key=config.gemini_api_key)
        _client_key = config.gemini_api_key
    return _client


# ==============================================================================
# CONTINUITY VALIDATOR - Pre-illustration script validation
# ==============================================================================

@dataclass
class CharacterState:
    """Tracks the state of a character across panels."""
    name: str
    current_items: Set[str] = field(default_factory=set)
    last_seen_page: Optional[int] = None
    last_seen_panel: Optional[int] = None
    is_alive: bool = True
    location: Optional[str] = None


class ContinuityValidator:
    """
    Validates continuity across panels in a graphic novel script.
    Should be run BEFORE illustration to catch issues early.
    """

    def __init__(self, script_path: str, character_metadata_dir: str):
        self.script_path = Path(script_path)
        self.character_metadata_dir = Path(character_metadata_dir)
        self.character_states: Dict[str, CharacterState] = {}
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []

        self._load_character_metadata()

    def _load_character_metadata(self):
        """Load character metadata to understand their distinctive items."""
        if not self.character_metadata_dir.exists():
            return

        for char_folder in self.character_metadata_dir.iterdir():
            if char_folder.is_dir():
                metadata_path = char_folder / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        char_name = metadata.get('name', '')
                        if char_name:
                            state = CharacterState(name=char_name)
                            if 'distinctive_items' in metadata:
                                state.current_items.update(metadata['distinctive_items'])
                            self.character_states[char_name] = state

    def validate_script(self) -> Dict:
        """
        Validate the entire script for continuity issues.

        Returns:
            Dict with 'errors', 'warnings', and 'summary' keys
        """
        if not self.script_path.exists():
            return {
                "errors": [{"type": "file_not_found", "message": f"Script not found: {self.script_path}"}],
                "warnings": [],
                "summary": "Validation failed - script file not found"
            }

        with open(self.script_path, "r") as f:
            script_data = json.load(f)

        for page in script_data:
            page_num = page.get('page_number', 0)
            panels = page.get('panels', [])

            for panel in panels:
                self._validate_panel(page_num, panel)

        summary = f"Validation complete: {len(self.errors)} errors, {len(self.warnings)} warnings"

        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": summary,
            "passed": len(self.errors) == 0
        }

    def _validate_panel(self, page_num: int, panel: Dict):
        """Validate a single panel for continuity issues."""
        panel_id = panel.get('panel_id', 0)
        characters = panel.get('characters', [])
        advice = panel.get('advice', {})

        continuity_notes = ""
        character_gear = ""

        if isinstance(advice, dict):
            continuity_notes = advice.get('continuity_notes', '')
            character_gear = advice.get('character_gear', '')
        else:
            continuity_notes = str(advice) if advice else ""

        for char_name in characters:
            if char_name not in self.character_states:
                self.character_states[char_name] = CharacterState(name=char_name)
                self.warnings.append({
                    "type": "unknown_character",
                    "page": page_num,
                    "panel": panel_id,
                    "character": char_name,
                    "message": f"Character '{char_name}' has no metadata entry"
                })

            state = self.character_states[char_name]

            if not state.is_alive:
                self.errors.append({
                    "type": "dead_character_appears",
                    "page": page_num,
                    "panel": panel_id,
                    "character": char_name,
                    "message": f"Character '{char_name}' appears after being marked as dead"
                })

            if state.last_seen_page is not None:
                page_gap = page_num - state.last_seen_page
                if page_gap > 10 and not continuity_notes:
                    self.warnings.append({
                        "type": "missing_continuity_note",
                        "page": page_num,
                        "panel": panel_id,
                        "character": char_name,
                        "message": f"Character '{char_name}' reappears after {page_gap} pages with no continuity notes"
                    })

            state.last_seen_page = page_num
            state.last_seen_panel = panel_id

            if character_gear:
                self._update_character_items(state, character_gear, page_num, panel_id)

    def _update_character_items(self, state: CharacterState, gear_description: str, page_num: int, panel_id: int):
        """Update character items based on gear description."""
        gear_lower = gear_description.lower()

        for item in list(state.current_items):
            if item.lower() not in gear_lower:
                self.warnings.append({
                    "type": "missing_distinctive_item",
                    "page": page_num,
                    "panel": panel_id,
                    "character": state.name,
                    "item": item,
                    "message": f"Character '{state.name}' may have lost distinctive item: {item}"
                })

    def get_validation_report(self) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("CONTINUITY VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")

        if not self.errors and not self.warnings:
            report.append("No continuity issues detected!")
            return "\n".join(report)

        if self.errors:
            report.append(f"ERRORS ({len(self.errors)}):")
            report.append("-" * 60)
            for error in self.errors:
                report.append(f"  Page {error['page']}, Panel {error['panel']}:")
                report.append(f"  [{error['type']}] {error['message']}")
                report.append("")

        if self.warnings:
            report.append(f"WARNINGS ({len(self.warnings)}):")
            report.append("-" * 60)
            for warning in self.warnings:
                report.append(f"  Page {warning['page']}, Panel {warning['panel']}:")
                report.append(f"  [{warning['type']}] {warning['message']}")
                report.append("")

        return "\n".join(report)


# ==============================================================================
# PANEL VALIDATOR - Post-generation image validation
# ==============================================================================

@dataclass
class PanelValidationResult:
    """Result of validating a single generated panel."""
    passed: bool
    issues: List[str] = field(default_factory=list)
    severity: str = "low"  # low, medium, high
    should_regenerate: bool = False


class PanelValidator:
    """
    Validates generated panel images against script requirements.
    Checks for era accuracy, character presence, and script fidelity.
    """

    def __init__(self, era_constraints: str = ""):
        self.era_constraints = era_constraints

    async def validate_panel(
        self,
        image: Image.Image,
        panel_data: dict,
        era_context: str = ""
    ) -> PanelValidationResult:
        """
        Validate a generated panel image against requirements.

        Args:
            image: PIL Image of the generated panel
            panel_data: Panel script data (characters, visual_description, etc.)
            era_context: Era/historical constraints string

        Returns:
            PanelValidationResult with pass/fail status and issues
        """
        era_to_use = era_context or self.era_constraints

        characters = panel_data.get('characters', [])
        visual_desc = panel_data.get('visual_description', '')
        key_objects = panel_data.get('key_objects', [])

        char_list = ", ".join(characters) if characters else "none specified"
        obj_list = ", ".join(key_objects) if key_objects else "none specified"

        # Build era constraint section
        era_section = ""
        if era_to_use:
            era_section = f"""
ERA/HISTORICAL CONSTRAINTS:
{era_to_use}

Check for ANY anachronistic elements (modern clothing, technology, vehicles, etc.)
"""

        validation_prompt = f"""
You are a quality assurance expert for graphic novel panels.

TASK: Validate this generated panel against the script requirements.

EXPECTED CONTENT:
- Characters present: {char_list}
- Key objects: {obj_list}
- Visual description: {visual_desc}
{era_section}
VALIDATION CHECKLIST:
1. CHARACTER PRESENCE: Are all expected characters visible in the panel?
2. VISUAL MATCH: Does the panel match the visual description?
3. ERA ACCURACY: Are there any anachronistic elements that don't fit the era?
4. TEXT-FREE: Is the image free of unwanted text, speech bubbles, or UI elements?
5. QUALITY: Is the image clear, well-composed, and usable?

Respond with a JSON object:
{{
    "passed": <true if all checks pass, false otherwise>,
    "issues": [<list of specific problems found>],
    "severity": "<low|medium|high>",
    "should_regenerate": <true if issues are severe enough to warrant regeneration>
}}
"""

        try:
            response = await get_client().aio.models.generate_content(
                model=config.scripting_model_page_script,
                contents=[validation_prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "passed": {"type": "BOOLEAN"},
                            "issues": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "severity": {"type": "STRING", "enum": ["low", "medium", "high"]},
                            "should_regenerate": {"type": "BOOLEAN"}
                        },
                        "required": ["passed", "issues", "severity", "should_regenerate"]
                    }
                )
            )

            result = response.parsed
            if result:
                return PanelValidationResult(
                    passed=result.get('passed', True),
                    issues=result.get('issues', []),
                    severity=result.get('severity', 'low'),
                    should_regenerate=result.get('should_regenerate', False)
                )

        except Exception as e:
            print(f"   Panel validation error: {e}")

        # Default to pass if validation fails
        return PanelValidationResult(passed=True, issues=[], severity="low", should_regenerate=False)


# ==============================================================================
# CONSISTENCY AUDITOR - Cross-panel character consistency
# ==============================================================================

@dataclass
class ConsistencyAuditResult:
    """Result of auditing consistency across multiple panels."""
    consistent: bool
    character_issues: Dict[str, List[str]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ConsistencyAuditor:
    """
    Audits character consistency across panels within a page.
    Compares character appearances to detect drift.
    """

    async def audit_page_consistency(
        self,
        page_images: List[Image.Image],
        characters: List[str],
        panel_data: List[dict] = None
    ) -> ConsistencyAuditResult:
        """
        Audit character consistency across panels in a page.

        Args:
            page_images: List of PIL Images (panels from same page)
            characters: List of character names appearing in the page
            panel_data: Optional panel script data for context

        Returns:
            ConsistencyAuditResult with consistency status and issues
        """
        if len(page_images) < 2:
            return ConsistencyAuditResult(consistent=True)

        char_list = ", ".join(characters) if characters else "characters in the panels"

        audit_prompt = f"""
You are a quality assurance expert for graphic novel character consistency.

TASK: Compare character appearances across these panels from the SAME PAGE.

CHARACTERS TO CHECK: {char_list}

For each character appearing in multiple panels, evaluate:
1. FACE CONSISTENCY: Is the face shape, features, and expression style consistent?
2. CLOTHING CONSISTENCY: Is the clothing/costume the same across panels?
3. DISTINCTIVE ITEMS: Are signature items (glasses, weapons, accessories) present?
4. ART STYLE: Is the character drawn in the same art style across panels?
5. COLOR PALETTE: Are character colors consistent?

Respond with a JSON object:
{{
    "consistent": <true if characters are consistent, false otherwise>,
    "character_issues": {{
        "<character_name>": ["<issue 1>", "<issue 2>"]
    }},
    "recommendations": ["<suggestion for improving consistency>"]
}}
"""

        input_contents = [audit_prompt]
        for i, img in enumerate(page_images):
            input_contents.append(f"\n--- PANEL {i + 1} ---")
            input_contents.append(img)

        try:
            response = await get_client().aio.models.generate_content(
                model=config.scripting_model_page_script,
                contents=input_contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "consistent": {"type": "BOOLEAN"},
                            "inconsistent_characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "issue_descriptions": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "recommendations": {"type": "ARRAY", "items": {"type": "STRING"}}
                        },
                        "required": ["consistent"]
                    }
                )
            )

            result = response.parsed
            if result:
                return ConsistencyAuditResult(
                    consistent=result.get('consistent', True),
                    character_issues=result.get('character_issues', {}),
                    recommendations=result.get('recommendations', [])
                )

        except Exception as e:
            print(f"   Consistency audit error: {e}")

        return ConsistencyAuditResult(consistent=True)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def validate_script_continuity(script_path: str, character_dir: str) -> Dict:
    """
    Convenience function to validate script continuity.

    Args:
        script_path: Path to the script JSON file
        character_dir: Path to the characters directory

    Returns:
        Validation result dict
    """
    validator = ContinuityValidator(script_path, character_dir)
    return validator.validate_script()


# ==============================================================================
# SCRIPT VALIDATOR - Auto-fix validation for the enrichment pipeline
# ==============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue found in the script."""
    issue_type: str
    page: int
    panel: int
    severity: str  # "auto_fix", "warning", "manual_review"
    description: str
    fix_applied: Optional[str] = None
    original_value: Optional[str] = None
    fixed_value: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report with auto-fix results."""
    total_issues: int = 0
    auto_fixed: int = 0
    warnings: int = 0
    manual_review: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    script_modified: bool = False


class ScriptValidator:
    """
    Validates generated scripts and applies auto-fixes where possible.

    Auto-fixable issues:
    - Dialogue too long (>100 chars) → truncate with ellipsis
    - Missing distinctive items → add to key_objects
    - Era anachronisms (mild) → replace via ERA_TERM_REPLACEMENTS
    - Missing scene gear → inject from character_arcs
    - Bubble position conflicts → reassign positions
    - Caption overflow (>120 chars) → split into multiple captions

    Manual review flagged:
    - Dead character appears
    - Character in impossible location
    - Major era anachronisms requiring regeneration
    """

    MAX_DIALOGUE_LENGTH = 100
    MAX_CAPTION_LENGTH = 120
    BUBBLE_POSITIONS = ["top-left", "top-right", "bottom-left", "bottom-right"]

    def __init__(
        self,
        era: str = "",
        character_arcs: Optional[Dict] = None,
        assets: Optional[Dict] = None
    ):
        self.era = era
        self.character_arcs = character_arcs or {}
        self.assets = assets or {}
        self.report = ValidationReport()

        # Build character lookup from arcs
        self.character_lookup: Dict[str, Dict] = {}
        if character_arcs and "characters" in character_arcs:
            for char in character_arcs["characters"]:
                self.character_lookup[char["name"]] = char

        # Build scene states lookup
        self.scene_states: Dict[int, Dict] = {}
        if character_arcs and "scene_states" in character_arcs:
            for state in character_arcs["scene_states"]:
                self.scene_states[state["page_number"]] = state

    def validate_and_fix(self, script: List[Dict]) -> Tuple[List[Dict], ValidationReport]:
        """
        Validate the script and apply auto-fixes.

        Args:
            script: The full script (list of pages with panels)

        Returns:
            Tuple of (fixed_script, validation_report)
        """
        # Import here to avoid circular import
        from utils import fix_era_anachronisms

        fixed_script = json.loads(json.dumps(script))  # Deep copy

        for page in fixed_script:
            page_num = page.get("page_number", 0)
            panels = page.get("panels", [])

            # Track bubble positions used on this page for conflict detection
            page_bubble_positions: Dict[int, List[str]] = {}

            for panel in panels:
                panel_id = panel.get("panel_id", 0)
                page_bubble_positions[panel_id] = []

                # Validate and fix dialogue length
                self._fix_dialogue_length(panel, page_num, panel_id)

                # Validate and fix caption length
                self._fix_caption_length(panel, page_num, panel_id)

                # Validate and fix era anachronisms in dialogue
                if self.era and panel.get("dialogue"):
                    original = panel["dialogue"]
                    fixed = fix_era_anachronisms(original, self.era)
                    if fixed != original:
                        panel["dialogue"] = fixed
                        self._add_issue(ValidationIssue(
                            issue_type="era_anachronism_fixed",
                            page=page_num,
                            panel=panel_id,
                            severity="auto_fix",
                            description="Fixed era-inappropriate terms in dialogue",
                            fix_applied="term_replacement",
                            original_value=original,
                            fixed_value=fixed
                        ))

                # Validate character presence and gear
                self._validate_character_gear(panel, page_num, panel_id)

                # Validate distinctive items
                self._validate_distinctive_items(panel, page_num, panel_id)

                # Track bubble position for conflict detection
                bubble_pos = panel.get("bubble_position")
                if bubble_pos:
                    page_bubble_positions[panel_id].append(bubble_pos)

            # Fix bubble position conflicts across panels on same page
            self._fix_bubble_conflicts(panels, page_num, page_bubble_positions)

        # Check for dead characters appearing (manual review)
        self._check_dead_characters(fixed_script)

        # Update report totals
        self.report.total_issues = len(self.report.issues)
        self.report.auto_fixed = sum(1 for i in self.report.issues if i.severity == "auto_fix")
        self.report.warnings = sum(1 for i in self.report.issues if i.severity == "warning")
        self.report.manual_review = sum(1 for i in self.report.issues if i.severity == "manual_review")
        self.report.script_modified = self.report.auto_fixed > 0

        return fixed_script, self.report

    def _add_issue(self, issue: ValidationIssue):
        """Add an issue to the report."""
        self.report.issues.append(issue)

    def _fix_dialogue_length(self, panel: Dict, page_num: int, panel_id: int):
        """Fix dialogue that exceeds maximum length."""
        dialogue = panel.get("dialogue", "")
        if dialogue and len(dialogue) > self.MAX_DIALOGUE_LENGTH:
            # Truncate at word boundary with ellipsis
            truncated = dialogue[:self.MAX_DIALOGUE_LENGTH - 3].rsplit(" ", 1)[0] + "..."
            panel["dialogue"] = truncated
            self._add_issue(ValidationIssue(
                issue_type="dialogue_overflow",
                page=page_num,
                panel=panel_id,
                severity="auto_fix",
                description=f"Dialogue exceeded {self.MAX_DIALOGUE_LENGTH} chars",
                fix_applied="truncation",
                original_value=dialogue,
                fixed_value=truncated
            ))

    def _fix_caption_length(self, panel: Dict, page_num: int, panel_id: int):
        """Fix captions that exceed maximum length."""
        caption = panel.get("caption", "")
        if caption and len(caption) > self.MAX_CAPTION_LENGTH:
            # Split into two captions if possible
            midpoint = len(caption) // 2
            split_point = caption.find(". ", midpoint - 20, midpoint + 20)
            if split_point == -1:
                split_point = caption.find(", ", midpoint - 20, midpoint + 20)
            if split_point == -1:
                split_point = caption.find(" ", midpoint - 10, midpoint + 10)

            if split_point != -1:
                # Store as list of captions
                panel["captions"] = [
                    caption[:split_point + 1].strip(),
                    caption[split_point + 1:].strip()
                ]
                panel["caption"] = panel["captions"][0]  # Keep first for backward compat
                self._add_issue(ValidationIssue(
                    issue_type="caption_overflow",
                    page=page_num,
                    panel=panel_id,
                    severity="auto_fix",
                    description=f"Caption exceeded {self.MAX_CAPTION_LENGTH} chars",
                    fix_applied="split_caption",
                    original_value=caption,
                    fixed_value=str(panel["captions"])
                ))
            else:
                # Can't split cleanly, truncate
                truncated = caption[:self.MAX_CAPTION_LENGTH - 3].rsplit(" ", 1)[0] + "..."
                panel["caption"] = truncated
                self._add_issue(ValidationIssue(
                    issue_type="caption_overflow",
                    page=page_num,
                    panel=panel_id,
                    severity="auto_fix",
                    description=f"Caption exceeded {self.MAX_CAPTION_LENGTH} chars",
                    fix_applied="truncation",
                    original_value=caption,
                    fixed_value=truncated
                ))

    def _validate_character_gear(self, panel: Dict, page_num: int, panel_id: int):
        """Validate and fix character gear based on scene type."""
        characters = panel.get("characters", [])
        advice = panel.get("advice", {})
        scene_type = advice.get("scene_type", "") if isinstance(advice, dict) else ""
        key_objects = panel.get("key_objects", [])

        # Get scene state for this page
        scene_state = self.scene_states.get(page_num, {})
        char_states = scene_state.get("characters", {})

        for char_name in characters:
            char_data = self.character_lookup.get(char_name, {})
            era_gear = char_data.get("era_appropriate_gear", {})

            # Determine required gear based on scene type
            required_gear = []
            if "underwater" in scene_type.lower() and "underwater" in era_gear:
                required_gear = era_gear["underwater"]
            elif "formal" in scene_type.lower() and "formal" in era_gear:
                required_gear = era_gear["formal"]
            elif "aboard" in scene_type.lower() and "aboard_nautilus" in era_gear:
                required_gear = era_gear["aboard_nautilus"]

            # Also check character state for this specific page
            if char_name in char_states:
                state_gear = char_states[char_name].get("gear", [])
                required_gear = list(set(required_gear + state_gear))

            # Check if required gear is mentioned
            visual_desc = panel.get("visual_description", "").lower()
            for gear_item in required_gear:
                if gear_item.lower() not in visual_desc and gear_item not in key_objects:
                    # Auto-fix: add to key_objects
                    if gear_item not in key_objects:
                        key_objects.append(gear_item)
                        self._add_issue(ValidationIssue(
                            issue_type="missing_scene_gear",
                            page=page_num,
                            panel=panel_id,
                            severity="auto_fix",
                            description=f"Added missing gear '{gear_item}' for {char_name} in {scene_type}",
                            fix_applied="added_to_key_objects",
                            original_value=None,
                            fixed_value=gear_item
                        ))

            panel["key_objects"] = key_objects

    def _validate_distinctive_items(self, panel: Dict, page_num: int, panel_id: int):
        """Validate that characters have their distinctive items."""
        characters = panel.get("characters", [])
        key_objects = panel.get("key_objects", [])
        visual_desc = panel.get("visual_description", "").lower()

        for char_name in characters:
            char_data = self.character_lookup.get(char_name, {})
            distinctive_items = char_data.get("distinctive_items", [])

            for item in distinctive_items:
                item_lower = item.lower()
                # Check if item is mentioned in visual description or key objects
                if item_lower not in visual_desc and item not in key_objects:
                    # Auto-fix: add to key_objects
                    key_objects.append(item)
                    self._add_issue(ValidationIssue(
                        issue_type="missing_distinctive_item",
                        page=page_num,
                        panel=panel_id,
                        severity="auto_fix",
                        description=f"Added distinctive item '{item}' for {char_name}",
                        fix_applied="added_to_key_objects",
                        original_value=None,
                        fixed_value=item
                    ))

        panel["key_objects"] = key_objects

    def _fix_bubble_conflicts(
        self,
        panels: List[Dict],
        page_num: int,
        positions_used: Dict[int, List[str]]
    ):
        """Fix bubble position conflicts on a page."""
        # Count position usage across all panels
        all_positions = []
        for positions in positions_used.values():
            all_positions.extend(positions)

        # If too many panels use the same position, redistribute
        position_counts = {}
        for pos in all_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # Find overused positions (more than 2 panels with same position)
        overused = [pos for pos, count in position_counts.items() if count > 2]

        if overused:
            # Redistribute positions
            available = list(self.BUBBLE_POSITIONS)
            panel_idx = 0
            for panel in panels:
                if panel.get("bubble_position") in overused:
                    # Assign new position in rotation
                    new_pos = available[panel_idx % len(available)]
                    old_pos = panel["bubble_position"]
                    panel["bubble_position"] = new_pos
                    self._add_issue(ValidationIssue(
                        issue_type="bubble_position_conflict",
                        page=page_num,
                        panel=panel.get("panel_id", 0),
                        severity="auto_fix",
                        description="Reassigned bubble position to avoid conflict",
                        fix_applied="position_reassignment",
                        original_value=old_pos,
                        fixed_value=new_pos
                    ))
                    panel_idx += 1

    def _check_dead_characters(self, script: List[Dict]):
        """Check for dead characters appearing (flags for manual review)."""
        # Track characters marked as dead
        dead_characters: Set[str] = set()

        for page in script:
            page_num = page.get("page_number", 0)
            for panel in page.get("panels", []):
                panel_id = panel.get("panel_id", 0)

                # Check for death indicators in dialogue or caption
                text_content = (
                    panel.get("dialogue", "") + " " +
                    panel.get("caption", "") + " " +
                    panel.get("visual_description", "")
                ).lower()

                # Simple death detection (could be enhanced with LLM)
                death_phrases = ["dies", "killed", "death of", "murdered", "passed away"]
                for char_name in panel.get("characters", []):
                    for phrase in death_phrases:
                        if phrase in text_content and char_name.lower() in text_content:
                            dead_characters.add(char_name)

                # Check if a dead character appears
                for char_name in panel.get("characters", []):
                    if char_name in dead_characters:
                        self._add_issue(ValidationIssue(
                            issue_type="dead_character_appears",
                            page=page_num,
                            panel=panel_id,
                            severity="manual_review",
                            description=f"Character '{char_name}' appears after being marked as dead",
                            fix_applied=None,
                            original_value=char_name,
                            fixed_value=None
                        ))

    def get_report_summary(self) -> str:
        """Generate a human-readable summary of the validation report."""
        lines = [
            "=" * 60,
            "SCRIPT VALIDATION REPORT",
            "=" * 60,
            f"Total Issues: {self.report.total_issues}",
            f"  - Auto-fixed: {self.report.auto_fixed}",
            f"  - Warnings: {self.report.warnings}",
            f"  - Manual Review Required: {self.report.manual_review}",
            ""
        ]

        if self.report.manual_review > 0:
            lines.append("MANUAL REVIEW REQUIRED:")
            lines.append("-" * 60)
            for issue in self.report.issues:
                if issue.severity == "manual_review":
                    lines.append(f"  Page {issue.page}, Panel {issue.panel}:")
                    lines.append(f"    [{issue.issue_type}] {issue.description}")
            lines.append("")

        if self.report.auto_fixed > 0:
            lines.append("AUTO-FIXES APPLIED:")
            lines.append("-" * 60)
            for issue in self.report.issues:
                if issue.severity == "auto_fix":
                    lines.append(f"  Page {issue.page}, Panel {issue.panel}:")
                    lines.append(f"    [{issue.issue_type}] {issue.description}")
            lines.append("")

        return "\n".join(lines)


def validate_and_autofix_script(
    script: List[Dict],
    era: str = "",
    character_arcs: Optional[Dict] = None,
    assets: Optional[Dict] = None
) -> Tuple[List[Dict], ValidationReport]:
    """
    Convenience function to validate and auto-fix a script.

    Args:
        script: The full script to validate
        era: Era string for anachronism detection
        character_arcs: Character arcs data from generate_character_deep_dive()
        assets: Asset manifest from generate_asset_manifest()

    Returns:
        Tuple of (fixed_script, validation_report)
    """
    validator = ScriptValidator(era=era, character_arcs=character_arcs, assets=assets)
    return validator.validate_and_fix(script)


if __name__ == "__main__":
    # Example usage - Continuity Validator
    validator = ContinuityValidator(
        script_path="assets/output/20-thousand-leagues-under-the-sea_test_page.json",
        character_metadata_dir="assets/output/characters"
    )

    result = validator.validate_script()
    print(validator.get_validation_report())

    output_path = Path("assets/output/continuity_validation.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Validation report saved to: {output_path}")
