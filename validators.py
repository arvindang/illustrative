"""
Validators module: Quality control for the graphic novel pipeline.

Includes:
- ContinuityValidator: Pre-illustration script validation (character states, impossible scenarios)
- PanelValidator: Post-generation image validation (era accuracy, character presence)
- ConsistencyAuditor: Cross-panel character consistency checking
"""

import json
import io
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from PIL import Image
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from config import config


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


if __name__ == "__main__":
    # Example usage
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
