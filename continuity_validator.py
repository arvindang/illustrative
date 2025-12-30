"""
Continuity Validator: Tracks character states and detects impossible scenarios across panels.

This module provides validation for:
- Character state tracking (props, clothing, positions)
- Cross-panel continuity checking
- Impossible scenario detection
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional


class CharacterState:
    """Tracks the state of a character across panels."""

    def __init__(self, name: str):
        self.name = name
        self.current_items: Set[str] = set()  # Items character is holding/wearing
        self.last_seen_page: Optional[int] = None
        self.last_seen_panel: Optional[int] = None
        self.is_alive: bool = True
        self.location: Optional[str] = None  # General location description


class ContinuityValidator:
    """Validates continuity across panels in a graphic novel script."""

    def __init__(self, script_path: str, character_metadata_dir: str):
        self.script_path = Path(script_path)
        self.character_metadata_dir = Path(character_metadata_dir)
        self.character_states: Dict[str, CharacterState] = {}
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []

        # Load character metadata to understand their distinctive items
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
                        char_name = metadata['name']
                        state = CharacterState(char_name)
                        # Initialize with distinctive items from metadata
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

        # Validate each page
        for page in script_data:
            page_num = page.get('page_number', 0)
            panels = page.get('panels', [])

            for panel in panels:
                self._validate_panel(page_num, panel)

        # Generate summary
        summary = f"Validation complete: {len(self.errors)} errors, {len(self.warnings)} warnings"

        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": summary
        }

    def _validate_panel(self, page_num: int, panel: Dict):
        """Validate a single panel for continuity issues."""
        panel_id = panel.get('panel_id', 0)
        characters = panel.get('characters', [])
        advice = panel.get('advice', {})

        # Extract continuity notes if available
        continuity_notes = ""
        character_gear = ""

        if isinstance(advice, dict):
            continuity_notes = advice.get('continuity_notes', '')
            character_gear = advice.get('character_gear', '')
        else:
            # Backward compatibility - advice is a string
            continuity_notes = str(advice)

        # Check each character in the panel
        for char_name in characters:
            # Initialize character state if not exists
            if char_name not in self.character_states:
                self.character_states[char_name] = CharacterState(char_name)
                self.warnings.append({
                    "type": "unknown_character",
                    "page": page_num,
                    "panel": panel_id,
                    "character": char_name,
                    "message": f"Character '{char_name}' has no metadata entry"
                })

            state = self.character_states[char_name]

            # Check if character is alive
            if not state.is_alive:
                self.errors.append({
                    "type": "dead_character_appears",
                    "page": page_num,
                    "panel": panel_id,
                    "character": char_name,
                    "message": f"Character '{char_name}' appears after being marked as dead"
                })

            # Check for continuity with previous panel
            if state.last_seen_page is not None:
                # Character reappears - check if reasonable
                page_gap = page_num - state.last_seen_page

                if page_gap > 10 and not continuity_notes:
                    self.warnings.append({
                        "type": "missing_continuity_note",
                        "page": page_num,
                        "panel": panel_id,
                        "character": char_name,
                        "message": f"Character '{char_name}' reappears after {page_gap} pages with no continuity notes"
                    })

            # Update character state
            state.last_seen_page = page_num
            state.last_seen_panel = panel_id

            # Parse character gear from structured advice
            if character_gear:
                # Extract items mentioned in character_gear
                self._update_character_items(state, character_gear, page_num, panel_id)

    def _update_character_items(self, state: CharacterState, gear_description: str, page_num: int, panel_id: int):
        """Update character items based on gear description."""
        # Simple keyword extraction - could be enhanced with NLP
        # Look for items from distinctive_items
        gear_lower = gear_description.lower()

        # Check if character has lost their distinctive items
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

    def mark_character_dead(self, char_name: str, page_num: int, panel_id: int):
        """
        Manually mark a character as dead for validation purposes.

        This should be called when a character dies in the story.
        """
        if char_name in self.character_states:
            self.character_states[char_name].is_alive = False
            print(f"✅ Marked {char_name} as deceased at Page {page_num}, Panel {panel_id}")

    def get_validation_report(self) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("CONTINUITY VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")

        if not self.errors and not self.warnings:
            report.append("✅ No continuity issues detected!")
            report.append("")
            return "\n".join(report)

        if self.errors:
            report.append(f"🚨 ERRORS ({len(self.errors)}):")
            report.append("-" * 60)
            for error in self.errors:
                report.append(f"  Page {error['page']}, Panel {error['panel']}:")
                report.append(f"  [{error['type']}] {error['message']}")
                report.append("")

        if self.warnings:
            report.append(f"⚠️  WARNINGS ({len(self.warnings)}):")
            report.append("-" * 60)
            for warning in self.warnings:
                report.append(f"  Page {warning['page']}, Panel {warning['panel']}:")
                report.append(f"  [{warning['type']}] {warning['message']}")
                report.append("")

        report.append("=" * 60)
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    validator = ContinuityValidator(
        script_path="assets/output/20-thousand-leagues-under-the-sea_test_page.json",
        character_metadata_dir="assets/output/characters"
    )

    result = validator.validate_script()
    print(validator.get_validation_report())

    # Save validation report
    output_path = Path("assets/output/continuity_validation.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"📝 Validation report saved to: {output_path}")
