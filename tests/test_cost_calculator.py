"""
Unit tests for cost_calculator module.

Tests cover:
- Pricing constant validity
- Individual cost estimation functions
- Total cost estimation with various inputs
- Heuristic estimations
- Edge cases
"""
import pytest
from cost_calculator import (
    # Constants
    GEMINI_FLASH_INPUT_PER_M,
    GEMINI_FLASH_OUTPUT_PER_M,
    GEMINI_PRO_INPUT_PER_M,
    GEMINI_PRO_OUTPUT_PER_M,
    IMAGE_COST_1K_2K,
    IMAGE_COST_2K_3K,
    IMAGE_COST_4K,
    DEFAULT_MARGIN_PERCENT,
    PRESCAN_COST,
    AVG_PANELS_PER_PAGE,
    CHARACTER_REF_CANDIDATES,
    # Functions
    estimate_scripting_cost,
    estimate_image_cost,
    estimate_reference_cost,
    estimate_composition_cost,
    estimate_total_cost,
    estimate_characters_from_word_count,
    estimate_objects_from_word_count,
    quick_estimate,
    format_cost_summary,
    # Dataclasses
    CostEstimate,
    CostBreakdown,
)


class TestPricingConstants:
    """Test pricing constants are reasonable values."""

    def test_flash_model_prices(self):
        """Flash model prices should be cheaper than pro."""
        assert GEMINI_FLASH_INPUT_PER_M < GEMINI_PRO_INPUT_PER_M
        assert GEMINI_FLASH_OUTPUT_PER_M < GEMINI_PRO_OUTPUT_PER_M

    def test_image_costs_scale_with_resolution(self):
        """Higher resolution images should cost more."""
        assert IMAGE_COST_1K_2K < IMAGE_COST_2K_3K < IMAGE_COST_4K

    def test_margin_is_reasonable(self):
        """Margin should be between 0 and 50%."""
        assert 0 < DEFAULT_MARGIN_PERCENT <= 0.5

    def test_prescan_cost_is_minimal(self):
        """Prescan should cost less than $0.05."""
        assert PRESCAN_COST < 0.05


class TestScriptingCost:
    """Test scripting cost estimation."""

    def test_basic_scripting_cost(self):
        """Basic scripting cost calculation."""
        input_cost, output_cost, savings = estimate_scripting_cost(
            word_count=50000, page_count=80
        )

        assert input_cost > 0
        assert output_cost > 0
        assert savings > 0  # Cache should provide savings

    def test_more_words_costs_more(self):
        """More words should cost more."""
        cost_small = estimate_scripting_cost(10000, 20)
        cost_large = estimate_scripting_cost(100000, 150)

        # Total cost (input + output) should be higher for larger documents
        total_small = cost_small[0] + cost_small[1]
        total_large = cost_large[0] + cost_large[1]

        assert total_large > total_small

    def test_cache_savings_significant(self):
        """Cache savings should be a significant fraction of input cost."""
        input_cost, output_cost, savings = estimate_scripting_cost(
            word_count=80000, page_count=100
        )

        # Savings should be at least 30% of what full cost would have been
        full_input_cost = input_cost + savings
        assert savings > full_input_cost * 0.3


class TestImageCost:
    """Test image generation cost estimation."""

    def test_basic_image_cost(self):
        """Basic image cost calculation."""
        cost = estimate_image_cost(page_count=50, avg_panels_per_page=4)

        expected = 50 * 4 * IMAGE_COST_1K_2K
        assert cost == pytest.approx(expected)

    def test_zero_pages_zero_cost(self):
        """Zero pages should have zero cost."""
        cost = estimate_image_cost(page_count=0)
        assert cost == 0

    def test_custom_panels_per_page(self):
        """Custom panels per page should affect cost."""
        cost_3 = estimate_image_cost(50, avg_panels_per_page=3)
        cost_5 = estimate_image_cost(50, avg_panels_per_page=5)

        assert cost_5 > cost_3


class TestReferenceCost:
    """Test reference sheet cost estimation."""

    def test_basic_reference_cost(self):
        """Basic reference cost calculation."""
        char_cost, obj_cost = estimate_reference_cost(
            num_characters=5, num_objects=3
        )

        expected_char = 5 * CHARACTER_REF_CANDIDATES * IMAGE_COST_1K_2K
        expected_obj = 3 * 1 * IMAGE_COST_1K_2K

        assert char_cost == pytest.approx(expected_char)
        assert obj_cost == pytest.approx(expected_obj)

    def test_zero_entities(self):
        """Zero characters/objects should have zero cost."""
        char_cost, obj_cost = estimate_reference_cost(0, 0)
        assert char_cost == 0
        assert obj_cost == 0

    def test_character_refs_more_expensive(self):
        """Characters cost more due to multiple candidates."""
        char_cost, obj_cost = estimate_reference_cost(1, 1)
        assert char_cost > obj_cost  # 3 candidates vs 1


class TestCompositionCost:
    """Test composition analysis cost estimation."""

    def test_basic_composition_cost(self):
        """Basic composition cost calculation."""
        cost = estimate_composition_cost(page_count=50, avg_panels_per_page=4)
        assert cost > 0

    def test_scales_with_panels(self):
        """Cost should scale with total panels."""
        cost_small = estimate_composition_cost(10, 4)
        cost_large = estimate_composition_cost(100, 4)

        assert cost_large > cost_small * 5  # Should be ~10x


class TestTotalCost:
    """Test total cost estimation."""

    def test_basic_total_cost(self):
        """Basic total cost calculation."""
        estimate = estimate_total_cost(
            word_count=50000,
            page_count=80,
            num_characters=6,
            num_objects=4
        )

        assert isinstance(estimate, CostEstimate)
        assert estimate.total_cost > 0
        assert estimate.margin_fee > 0
        assert estimate.subtotal_vertex > 0

    def test_margin_applied(self):
        """Margin should be correctly applied."""
        estimate = estimate_total_cost(
            word_count=50000,
            page_count=80,
            num_characters=5,
            num_objects=3,
            margin_percent=0.10
        )

        expected_margin = estimate.subtotal_vertex * 0.10
        assert estimate.margin_fee == pytest.approx(expected_margin, rel=0.01)
        assert estimate.total_cost == pytest.approx(
            estimate.subtotal_vertex + estimate.margin_fee, rel=0.01
        )

    def test_custom_margin(self):
        """Custom margin percentage should be respected."""
        estimate_10 = estimate_total_cost(50000, 80, 5, 3, margin_percent=0.10)
        estimate_20 = estimate_total_cost(50000, 80, 5, 3, margin_percent=0.20)

        assert estimate_20.margin_fee > estimate_10.margin_fee
        assert estimate_20.margin_percent == 0.20

    def test_prescan_included(self):
        """Prescan cost should be included when specified."""
        with_prescan = estimate_total_cost(
            50000, 80, 5, 3, include_prescan=True
        )
        without_prescan = estimate_total_cost(
            50000, 80, 5, 3, include_prescan=False
        )

        assert with_prescan.breakdown.prescan > 0
        assert without_prescan.breakdown.prescan == 0
        assert with_prescan.subtotal_vertex > without_prescan.subtotal_vertex

    def test_counts_stored(self):
        """Counts should be stored in the estimate."""
        estimate = estimate_total_cost(
            word_count=50000,
            page_count=80,
            num_characters=6,
            num_objects=4,
            avg_panels_per_page=5
        )

        assert estimate.word_count == 50000
        assert estimate.pages == 80
        assert estimate.panels == 80 * 5
        assert estimate.characters == 6
        assert estimate.objects == 4


class TestHeuristics:
    """Test heuristic estimations."""

    def test_character_heuristic_short_story(self):
        """Short stories should have fewer characters."""
        chars = estimate_characters_from_word_count(10000)
        assert chars == 3

    def test_character_heuristic_novel(self):
        """Novels should have more characters."""
        chars = estimate_characters_from_word_count(80000)
        assert chars == 8

    def test_character_heuristic_epic(self):
        """Epics should have the most characters."""
        chars = estimate_characters_from_word_count(150000)
        assert chars == 12

    def test_object_heuristic_short_story(self):
        """Short stories should have fewer objects."""
        objs = estimate_objects_from_word_count(10000)
        assert objs == 2

    def test_object_heuristic_novel(self):
        """Novels should have more objects."""
        objs = estimate_objects_from_word_count(80000)
        assert objs == 6

    def test_characters_scale_with_length(self):
        """Character count should increase with word count."""
        chars_short = estimate_characters_from_word_count(5000)
        chars_long = estimate_characters_from_word_count(100000)

        assert chars_long > chars_short


class TestQuickEstimate:
    """Test quick estimate functionality."""

    def test_quick_estimate_returns_valid_result(self):
        """Quick estimate should return valid CostEstimate."""
        estimate = quick_estimate(word_count=50000)

        assert isinstance(estimate, CostEstimate)
        assert estimate.total_cost > 0
        assert estimate.pages > 0

    def test_quick_estimate_respects_page_override(self):
        """Quick estimate should respect page count override."""
        estimate = quick_estimate(word_count=50000, page_count=100)
        assert estimate.pages == 100

    def test_quick_estimate_no_prescan_cost(self):
        """Quick estimate should not include prescan cost."""
        estimate = quick_estimate(word_count=50000)
        assert estimate.breakdown.prescan == 0


class TestCostEstimateDataclass:
    """Test CostEstimate dataclass methods."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        estimate = estimate_total_cost(50000, 80, 5, 3)
        result = estimate.to_dict()

        assert isinstance(result, dict)
        assert "total_cost" in result
        assert "breakdown" in result
        assert isinstance(result["breakdown"], dict)

    def test_to_dict_rounding(self):
        """Dictionary values should be properly rounded."""
        estimate = estimate_total_cost(50000, 80, 5, 3)
        result = estimate.to_dict()

        # Total should be rounded to 2 decimal places
        total_str = str(result["total_cost"])
        if "." in total_str:
            decimals = len(total_str.split(".")[1])
            assert decimals <= 2


class TestFormatSummary:
    """Test cost summary formatting."""

    def test_format_produces_string(self):
        """Format should produce a non-empty string."""
        estimate = estimate_total_cost(50000, 80, 5, 3)
        summary = format_cost_summary(estimate)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_format_includes_key_values(self):
        """Summary should include key cost values."""
        estimate = estimate_total_cost(50000, 80, 5, 3)
        summary = format_cost_summary(estimate)

        # Should contain word count
        assert "50,000" in summary or "50000" in summary

        # Should contain total cost
        assert "Total" in summary


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_word_count(self):
        """Zero word count should still return valid estimate."""
        estimate = estimate_total_cost(
            word_count=0,
            page_count=1,
            num_characters=1,
            num_objects=1
        )

        assert estimate.total_cost >= 0

    def test_single_page(self):
        """Single page should work correctly."""
        estimate = estimate_total_cost(
            word_count=1000,
            page_count=1,
            num_characters=1,
            num_objects=1
        )

        assert estimate.pages == 1
        assert estimate.panels == AVG_PANELS_PER_PAGE

    def test_very_large_document(self):
        """Very large documents should work without overflow."""
        estimate = estimate_total_cost(
            word_count=500000,  # ~500k words
            page_count=200,
            num_characters=15,
            num_objects=10
        )

        assert estimate.total_cost > 0
        assert estimate.total_cost < 500  # Sanity check - should be under $500

    def test_zero_margin(self):
        """Zero margin should result in no margin fee."""
        estimate = estimate_total_cost(
            word_count=50000,
            page_count=80,
            num_characters=5,
            num_objects=3,
            margin_percent=0
        )

        assert estimate.margin_fee == 0
        assert estimate.total_cost == estimate.subtotal_vertex


class TestRealWorldExamples:
    """Test with real-world-like inputs.

    Image generation dominates costs at $0.134 per panel (1K-2K resolution).
    With 4 panels/page average, image cost = pages * 4 * $0.134.

    Expected costs:
    - 20 pages: 80 panels * $0.134 = $10.72 images + refs + scripting
    - 60 pages: 240 panels * $0.134 = $32.16 images + refs + scripting
    - 100 pages: 400 panels * $0.134 = $53.60 images + refs + scripting
    - 150 pages: 600 panels * $0.134 = $80.40 images + refs + scripting
    """

    def test_short_story_estimate(self):
        """Test short story (~10k words, 20 pages)."""
        estimate = estimate_total_cost(
            word_count=10000,
            page_count=20,
            num_characters=3,
            num_objects=2
        )

        # ~$10.72 for images, ~$1.47 for refs, minimal scripting
        # Total ~$13-15 with margin
        assert 10.0 < estimate.total_cost < 20.0
        # Verify image cost dominates
        assert estimate.breakdown.image_generation > estimate.breakdown.scripting_output

    def test_novella_estimate(self):
        """Test novella (~40k words, 60 pages)."""
        estimate = estimate_total_cost(
            word_count=40000,
            page_count=60,
            num_characters=5,
            num_objects=4
        )

        # ~$32.16 for images, ~$2.55 for refs, minimal scripting
        # Total ~$38-42 with margin
        assert 30.0 < estimate.total_cost < 50.0

    def test_novel_estimate(self):
        """Test novel (~80k words, 100 pages)."""
        estimate = estimate_total_cost(
            word_count=80000,
            page_count=100,
            num_characters=8,
            num_objects=6
        )

        # ~$53.60 for images, ~$4.02 for refs, minimal scripting
        # Total ~$63-70 with margin
        assert 55.0 < estimate.total_cost < 80.0

    def test_epic_estimate(self):
        """Test epic (~150k words, 150 pages)."""
        estimate = estimate_total_cost(
            word_count=150000,
            page_count=150,
            num_characters=12,
            num_objects=8
        )

        # ~$80.40 for images, ~$5.90 for refs, minimal scripting
        # Total ~$95-110 with margin
        assert 85.0 < estimate.total_cost < 120.0

    def test_cost_scales_linearly_with_pages(self):
        """Cost should scale roughly linearly with page count."""
        cost_50 = estimate_total_cost(50000, 50, 5, 4).total_cost
        cost_100 = estimate_total_cost(50000, 100, 5, 4).total_cost

        # Cost for 100 pages should be ~2x cost for 50 pages
        ratio = cost_100 / cost_50
        assert 1.8 < ratio < 2.2
