"""
Unit tests for validators (no API calls required).

Tests the validation logic in:
- PromptPreValidator: Pre-generation validation
- ScriptValidator: Post-generation validation with auto-fixes
- ContinuityValidator: Cross-panel continuity checking

Usage:
    # Run all validator tests
    pytest tests/test_validators.py -v

    # Run specific test class
    pytest tests/test_validators.py::TestPromptPreValidator -v
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validators import (
    PromptPreValidator,
    PreValidationResult,
    ScriptValidator,
    ValidationReport,
    ContinuityValidator,
    create_pre_validator_from_assets,
)


class TestPromptPreValidator:
    """Tests for PromptPreValidator (pre-generation validation)."""

    def test_dialogue_length_validation_pass(self):
        """Test that short dialogue passes validation."""
        validator = PromptPreValidator()
        result = validator.validate_panel_inputs(
            visual_description="A man standing in a room",
            dialogue="Hello there!"
        )
        assert result.valid is True
        assert len(result.issues) == 0

    def test_dialogue_length_validation_fail_and_autofix(self):
        """Test that long dialogue fails validation and gets auto-fixed."""
        validator = PromptPreValidator()
        long_dialogue = "A" * 100  # Exceeds 80 char limit
        result = validator.validate_panel_inputs(
            visual_description="A scene",
            dialogue=long_dialogue
        )
        assert result.valid is False
        assert "dialogue" in result.auto_fixed
        assert len(result.auto_fixed["dialogue"]) <= PromptPreValidator.MAX_DIALOGUE_CHARS

    def test_caption_length_validation_fail_and_autofix(self):
        """Test that long caption fails validation and gets auto-fixed."""
        validator = PromptPreValidator()
        long_caption = "B" * 120  # Exceeds 100 char limit
        result = validator.validate_panel_inputs(
            visual_description="A scene",
            caption=long_caption
        )
        assert result.valid is False
        assert "caption" in result.auto_fixed
        assert len(result.auto_fixed["caption"]) <= PromptPreValidator.MAX_CAPTION_CHARS

    def test_era_anachronism_detection_1860s(self):
        """Test detection of modern terms in 1860s era."""
        validator = PromptPreValidator(era="1860s Victorian")
        result = validator.validate_panel_inputs(
            visual_description="The man pulled out his phone and took a selfie"
        )
        assert result.valid is False
        assert any("phone" in issue.lower() for issue in result.issues)

    def test_era_anachronism_detection_medieval(self):
        """Test detection of modern terms in medieval era."""
        validator = PromptPreValidator(era="medieval")
        result = validator.validate_panel_inputs(
            visual_description="The knight fired his gun at the dragon"
        )
        assert result.valid is False
        assert any("gun" in issue.lower() for issue in result.issues)

    def test_era_appropriate_content_passes(self):
        """Test that era-appropriate content passes validation."""
        validator = PromptPreValidator(era="1860s")
        result = validator.validate_panel_inputs(
            visual_description="The captain adjusted his brass telescope and looked out to sea"
        )
        assert result.valid is True

    def test_dialogue_in_era_check(self):
        """Test that era check also applies to dialogue."""
        validator = PromptPreValidator(era="1860s")
        result = validator.validate_panel_inputs(
            visual_description="Two men talking",
            dialogue="Let me check my email real quick"
        )
        assert any("email" in issue.lower() for issue in result.issues)

    def test_unknown_character_warning(self):
        """Test that unknown characters generate suggestions."""
        validator = PromptPreValidator(
            known_characters=["Captain Nemo", "Professor Aronnax"]
        )
        result = validator.validate_panel_inputs(
            visual_description="A scene",
            characters=["Captain Nemo", "Unknown Person"]
        )
        assert any("unknown" in s.lower() for s in result.suggestions)

    def test_known_character_no_warning(self):
        """Test that known characters don't generate warnings."""
        validator = PromptPreValidator(
            known_characters=["Captain Nemo", "Professor Aronnax"]
        )
        result = validator.validate_panel_inputs(
            visual_description="A scene",
            characters=["Captain Nemo"]
        )
        assert not any("unknown" in s.lower() for s in result.suggestions)

    def test_preprocess_dialogue_removes_stage_directions(self):
        """Test that stage directions are removed from dialogue."""
        validator = PromptPreValidator()
        cleaned = validator.preprocess_dialogue("Hello (he waves) there!")
        assert "(he waves)" not in cleaned
        assert "Hello" in cleaned
        assert "there" in cleaned

    def test_preprocess_dialogue_fixes_punctuation(self):
        """Test that excessive punctuation is fixed."""
        validator = PromptPreValidator()
        assert validator.preprocess_dialogue("What?!") == "What?"
        assert validator.preprocess_dialogue("Stop!!!!") == "Stop!"
        assert validator.preprocess_dialogue("Really????") == "Really?"
        assert validator.preprocess_dialogue("Wait.....") == "Wait..."

    def test_truncate_at_word_boundary(self):
        """Test that truncation happens at word boundaries."""
        validator = PromptPreValidator()
        text = "This is a long sentence that needs to be truncated somewhere"
        truncated = validator._truncate_at_word_boundary(text, 30)
        assert len(truncated) <= 30
        assert truncated.endswith("...")
        # Should not cut in the middle of a word
        assert not truncated.endswith("sente...")


class TestScriptValidator:
    """Tests for ScriptValidator (post-generation validation with auto-fixes)."""

    def test_dialogue_overflow_autofix(self):
        """Test that long dialogue is auto-truncated."""
        validator = ScriptValidator()
        script = [{
            "page_number": 1,
            "panels": [{
                "panel_id": 1,
                "visual_description": "A scene",
                "dialogue": "X" * 150,  # Way over 100 char limit
                "characters": []
            }]
        }]
        fixed_script, report = validator.validate_and_fix(script)

        assert report.auto_fixed > 0
        assert len(fixed_script[0]["panels"][0]["dialogue"]) <= ScriptValidator.MAX_DIALOGUE_LENGTH
        assert any(i.issue_type == "dialogue_overflow" for i in report.issues)

    def test_caption_overflow_split(self):
        """Test that long captions are split when possible."""
        validator = ScriptValidator()
        script = [{
            "page_number": 1,
            "panels": [{
                "panel_id": 1,
                "visual_description": "A scene",
                "caption": "This is the first part of a long caption. This is the second part of the long caption that continues on.",
                "characters": []
            }]
        }]
        fixed_script, report = validator.validate_and_fix(script)

        # Should either split into multiple captions or truncate
        panel = fixed_script[0]["panels"][0]
        if "captions" in panel:
            assert len(panel["captions"]) == 2
        else:
            assert len(panel["caption"]) <= ScriptValidator.MAX_CAPTION_LENGTH

    def test_bubble_position_conflict_resolution(self):
        """Test that bubble position conflicts are resolved."""
        validator = ScriptValidator()
        # Three panels all with top-left bubbles (conflict)
        script = [{
            "page_number": 1,
            "panels": [
                {"panel_id": 1, "visual_description": "Scene 1", "bubble_position": "top-left", "characters": []},
                {"panel_id": 2, "visual_description": "Scene 2", "bubble_position": "top-left", "characters": []},
                {"panel_id": 3, "visual_description": "Scene 3", "bubble_position": "top-left", "characters": []},
            ]
        }]
        fixed_script, report = validator.validate_and_fix(script)

        # At least one should be reassigned
        positions = [p["bubble_position"] for p in fixed_script[0]["panels"]]
        assert len(set(positions)) > 1 or report.auto_fixed > 0

    def test_dead_character_detection(self):
        """Test that dead characters appearing later are flagged."""
        validator = ScriptValidator()
        script = [
            {
                "page_number": 1,
                "panels": [{
                    "panel_id": 1,
                    "visual_description": "John dies in the explosion",
                    "dialogue": "",
                    "caption": "John was killed instantly.",
                    "characters": ["John"]
                }]
            },
            {
                "page_number": 2,
                "panels": [{
                    "panel_id": 1,
                    "visual_description": "John waves hello",
                    "characters": ["John"]
                }]
            }
        ]
        fixed_script, report = validator.validate_and_fix(script)

        assert report.manual_review > 0
        assert any(i.issue_type == "dead_character_appears" for i in report.issues)

    def test_validation_report_summary(self):
        """Test that validation report generates a readable summary."""
        validator = ScriptValidator()
        script = [{
            "page_number": 1,
            "panels": [{
                "panel_id": 1,
                "visual_description": "A scene",
                "dialogue": "Y" * 150,
                "characters": []
            }]
        }]
        validator.validate_and_fix(script)
        summary = validator.get_report_summary()

        assert "SCRIPT VALIDATION REPORT" in summary
        assert "Total Issues" in summary

    def test_no_issues_clean_script(self):
        """Test that a clean script has no issues."""
        validator = ScriptValidator()
        script = [{
            "page_number": 1,
            "panels": [{
                "panel_id": 1,
                "visual_description": "A man sits at a desk",
                "dialogue": "Good morning.",
                "caption": "It was a quiet day.",
                "characters": ["Man"],
                "bubble_position": "top-left"
            }]
        }]
        fixed_script, report = validator.validate_and_fix(script)

        # May have warnings but should be minimal
        assert report.manual_review == 0


class TestContinuityValidator:
    """Tests for ContinuityValidator (cross-panel continuity)."""

    def test_missing_script_file(self):
        """Test handling of missing script file."""
        validator = ContinuityValidator(
            script_path="/nonexistent/path.json",
            character_metadata_dir="/nonexistent/chars"
        )
        result = validator.validate_script()

        assert result["passed"] is False
        assert any("not found" in e["message"].lower() for e in result["errors"])

    def test_unknown_character_warning(self):
        """Test that unknown characters generate warnings."""
        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            script = [{
                "page_number": 1,
                "panels": [{
                    "panel_id": 1,
                    "visual_description": "Scene",
                    "characters": ["Unknown Character"]
                }]
            }]
            json.dump(script, f)
            script_path = f.name

        # Create empty character dir
        with tempfile.TemporaryDirectory() as char_dir:
            validator = ContinuityValidator(script_path, char_dir)
            result = validator.validate_script()

            assert any(w["type"] == "unknown_character" for w in result["warnings"])

        # Cleanup
        Path(script_path).unlink()

    def test_character_reappearance_gap_warning(self):
        """Test warning when character reappears after many pages without notes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            script = [
                {
                    "page_number": 1,
                    "panels": [{"panel_id": 1, "characters": ["Hero"], "advice": {}}]
                },
                # Gap of pages 2-14
                {
                    "page_number": 15,
                    "panels": [{"panel_id": 1, "characters": ["Hero"], "advice": {}}]
                }
            ]
            json.dump(script, f)
            script_path = f.name

        with tempfile.TemporaryDirectory() as char_dir:
            validator = ContinuityValidator(script_path, char_dir)
            result = validator.validate_script()

            # Should warn about character reappearing after 14 pages with no continuity notes
            gap_warnings = [w for w in result["warnings"] if w["type"] == "missing_continuity_note"]
            assert len(gap_warnings) > 0

        Path(script_path).unlink()

    def test_validation_report_format(self):
        """Test that validation report is properly formatted."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            script = [{
                "page_number": 1,
                "panels": [{"panel_id": 1, "characters": ["Test"], "advice": {}}]
            }]
            json.dump(script, f)
            script_path = f.name

        with tempfile.TemporaryDirectory() as char_dir:
            validator = ContinuityValidator(script_path, char_dir)
            validator.validate_script()
            report = validator.get_validation_report()

            assert "CONTINUITY VALIDATION REPORT" in report
            assert "=" * 60 in report

        Path(script_path).unlink()


class TestValidatorFactory:
    """Tests for validator factory functions."""

    def test_create_pre_validator_from_assets(self):
        """Test creating a pre-validator from asset data."""
        assets = {
            "characters": [
                {"name": "Captain Nemo"},
                {"name": "Professor Aronnax"}
            ],
            "locations": [
                {"name": "The Nautilus"},
                {"name": "Underwater Cave"}
            ],
            "interaction_rules": {
                "underwater_scenes": ["Characters must wear diving suits"]
            }
        }

        validator = create_pre_validator_from_assets(
            era="1860s Victorian",
            assets=assets
        )

        assert "Captain Nemo" in validator.known_characters
        assert "Professor Aronnax" in validator.known_characters
        assert "The Nautilus" in validator.known_locations
        assert len(validator.forbidden_terms) > 0  # 1860s terms loaded

    def test_create_pre_validator_with_character_arcs(self):
        """Test creating pre-validator with character arcs data."""
        character_arcs = {
            "characters": [
                {"name": "Hero"},
                {"name": "Villain"}
            ]
        }

        validator = create_pre_validator_from_assets(
            era="medieval",
            character_arcs=character_arcs
        )

        assert "Hero" in validator.known_characters
        assert "Villain" in validator.known_characters

    def test_create_pre_validator_empty_inputs(self):
        """Test creating pre-validator with empty inputs."""
        validator = create_pre_validator_from_assets()

        assert len(validator.known_characters) == 0
        assert len(validator.known_locations) == 0
        assert validator.era == ""


class TestValidationEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_script_validation(self):
        """Test validating an empty script."""
        validator = ScriptValidator()
        fixed_script, report = validator.validate_and_fix([])

        assert fixed_script == []
        assert report.total_issues == 0

    def test_empty_panel_list(self):
        """Test validating a page with no panels."""
        validator = ScriptValidator()
        script = [{"page_number": 1, "panels": []}]
        fixed_script, report = validator.validate_and_fix(script)

        assert fixed_script[0]["panels"] == []
        assert report.total_issues == 0

    def test_missing_fields_handled_gracefully(self):
        """Test that missing fields don't crash validation."""
        validator = ScriptValidator()
        script = [{
            "page_number": 1,
            "panels": [{
                "panel_id": 1
                # Missing visual_description, dialogue, caption, characters
            }]
        }]
        # Should not raise an exception
        fixed_script, report = validator.validate_and_fix(script)
        assert fixed_script is not None

    def test_exactly_at_length_limit(self):
        """Test content exactly at the length limit passes."""
        validator = PromptPreValidator()
        dialogue_at_limit = "X" * PromptPreValidator.MAX_DIALOGUE_CHARS
        result = validator.validate_panel_inputs(
            visual_description="A scene",
            dialogue=dialogue_at_limit
        )
        assert result.valid is True
        assert "dialogue" not in result.auto_fixed

    def test_one_char_over_limit(self):
        """Test content one character over the limit fails."""
        validator = PromptPreValidator()
        dialogue_over_limit = "X" * (PromptPreValidator.MAX_DIALOGUE_CHARS + 1)
        result = validator.validate_panel_inputs(
            visual_description="A scene",
            dialogue=dialogue_over_limit
        )
        assert result.valid is False
        assert "dialogue" in result.auto_fixed


# Allow running as standalone script
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
