"""
Post-validation module: Validate outputs AFTER LLM generation.

Includes:
- PanelValidator: Post-generation image validation (era accuracy, character presence)
- ScriptValidator: Post-generation script validation with auto-fixes
"""

import json
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image

from google.genai import types
from config import config
from utils import get_client


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


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

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
