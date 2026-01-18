"""
Pre-validation module: Validate inputs BEFORE expensive LLM calls.

Includes:
- PromptPreValidator: PRE-generation validation to avoid wasting tokens
- ContinuityValidator: Pre-illustration script validation (character states, impossible scenarios)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


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
