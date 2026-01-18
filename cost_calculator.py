"""
Cost estimation for Vertex AI processing.

Calculates estimated costs for processing documents through the Illustrative AI
pipeline with a configurable margin fee for pay-per-use pricing.

Pricing Sources:
- https://cloud.google.com/vertex-ai/generative-ai/pricing
- https://ai.google.dev/gemini-api/docs/pricing
"""
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# PRICING CONSTANTS (per million tokens, as of Jan 2026)
# ============================================================================

# Gemini 2.5/3 Flash Text Models
GEMINI_FLASH_INPUT_PER_M = 0.15       # $/M input tokens
GEMINI_FLASH_OUTPUT_PER_M = 0.60     # $/M output tokens

# Gemini Pro Text Models (for reasoning passes)
GEMINI_PRO_INPUT_PER_M = 0.30        # $/M input tokens
GEMINI_PRO_OUTPUT_PER_M = 1.20       # $/M output tokens

# Context caching discount (90% savings on cached tokens)
CACHE_INPUT_DISCOUNT = 0.90

# Cache storage cost
CACHE_STORAGE_PER_M_PER_HOUR = 1.00  # $/M tokens/hour (usually negligible for our use)

# Image generation costs (based on output tokens/resolution)
IMAGE_COST_1K_2K = 0.134             # 1024-2048px, ~1,120 output tokens
IMAGE_COST_2K_3K = 0.18              # 2048-3072px HD
IMAGE_COST_4K = 0.24                 # 4096px, ~2,000 output tokens

# Default resolution for cost estimation
DEFAULT_IMAGE_COST = IMAGE_COST_1K_2K

# Pre-scan cost (lightweight LLM call for character extraction)
PRESCAN_COST = 0.01                  # ~$0.01 for quick scan

# Default margin percentage
DEFAULT_MARGIN_PERCENT = 0.10        # 10% fee


# ============================================================================
# PIPELINE USAGE ESTIMATES (based on observed token usage)
# ============================================================================

# Scripting Agent token estimates
# 7-pass pipeline with context caching
SCRIPTING_INPUT_MULTIPLIER = 0.25    # ~4 chars per token, rough estimate
SCRIPTING_OUTPUT_PER_PAGE = 5000     # ~5K output tokens per page
SCRIPTING_CACHED_FRACTION = 0.70     # ~70% of input is cached across passes

# Image generation estimates
AVG_PANELS_PER_PAGE = 4              # Average panels per page
CHARACTER_REF_CANDIDATES = 3         # 3 candidates per character reference
OBJECT_REF_CANDIDATES = 1            # 1 reference per object

# Composition analysis
COMPOSITION_TOKENS_PER_PANEL = 1000  # ~1K tokens for analyzing each panel


@dataclass
class CostBreakdown:
    """Detailed breakdown of cost components."""
    prescan: float
    scripting_input: float
    scripting_output: float
    cache_savings: float
    image_generation: float
    character_refs: float
    object_refs: float
    composition_analysis: float


@dataclass
class CostEstimate:
    """Complete cost estimate with all components."""
    # Counts
    word_count: int
    pages: int
    panels: int
    characters: int
    objects: int

    # Cost breakdown
    breakdown: CostBreakdown

    # Totals
    subtotal_vertex: float
    margin_fee: float
    margin_percent: float
    total_cost: float

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "word_count": self.word_count,
            "estimated_pages": self.pages,
            "estimated_panels": self.panels,
            "characters": self.characters,
            "objects": self.objects,
            "breakdown": {
                "prescan": round(self.breakdown.prescan, 4),
                "scripting_input": round(self.breakdown.scripting_input, 4),
                "scripting_output": round(self.breakdown.scripting_output, 4),
                "cache_savings": round(self.breakdown.cache_savings, 4),
                "image_generation": round(self.breakdown.image_generation, 4),
                "character_refs": round(self.breakdown.character_refs, 4),
                "object_refs": round(self.breakdown.object_refs, 4),
                "composition_analysis": round(self.breakdown.composition_analysis, 4),
            },
            "subtotal_vertex": round(self.subtotal_vertex, 4),
            "margin_fee": round(self.margin_fee, 4),
            "margin_percent": self.margin_percent,
            "total_cost": round(self.total_cost, 2),
        }


def estimate_scripting_cost(
    word_count: int,
    page_count: int,
    use_pro_model: bool = True
) -> tuple[float, float, float]:
    """
    Estimate cost for the scripting agent's 7-pass pipeline.

    Args:
        word_count: Number of words in source document
        page_count: Target page count for output
        use_pro_model: Whether pro model is used for some passes (higher cost)

    Returns:
        Tuple of (input_cost, output_cost, cache_savings)
    """
    # Calculate token estimates
    input_tokens = word_count * SCRIPTING_INPUT_MULTIPLIER
    output_tokens = page_count * SCRIPTING_OUTPUT_PER_PAGE

    # Calculate cached vs non-cached tokens
    cached_tokens = input_tokens * SCRIPTING_CACHED_FRACTION
    non_cached_tokens = input_tokens * (1 - SCRIPTING_CACHED_FRACTION)

    # Mix of pro (Passes 0-4) and flash (Passes 5-6) models
    # Roughly 60% pro, 40% flash for input
    if use_pro_model:
        pro_fraction = 0.60
        flash_fraction = 0.40
        input_rate = (pro_fraction * GEMINI_PRO_INPUT_PER_M +
                      flash_fraction * GEMINI_FLASH_INPUT_PER_M)
        output_rate = (pro_fraction * GEMINI_PRO_OUTPUT_PER_M +
                       flash_fraction * GEMINI_FLASH_OUTPUT_PER_M)
    else:
        input_rate = GEMINI_FLASH_INPUT_PER_M
        output_rate = GEMINI_FLASH_OUTPUT_PER_M

    # Cost for non-cached tokens (full price)
    non_cached_cost = (non_cached_tokens / 1_000_000) * input_rate

    # Cost for cached tokens (90% discount)
    cached_cost = (cached_tokens / 1_000_000) * input_rate * (1 - CACHE_INPUT_DISCOUNT)

    # Output tokens cost (no caching)
    output_cost = (output_tokens / 1_000_000) * output_rate

    # Calculate savings from caching
    savings = (cached_tokens / 1_000_000) * input_rate * CACHE_INPUT_DISCOUNT

    input_cost = non_cached_cost + cached_cost

    return input_cost, output_cost, savings


def estimate_image_cost(
    page_count: int,
    avg_panels_per_page: int = AVG_PANELS_PER_PAGE,
    cost_per_image: float = DEFAULT_IMAGE_COST
) -> float:
    """
    Estimate cost for panel image generation.

    Args:
        page_count: Number of pages
        avg_panels_per_page: Average panels per page
        cost_per_image: Cost per generated image

    Returns:
        Total image generation cost
    """
    total_panels = page_count * avg_panels_per_page
    return total_panels * cost_per_image


def estimate_reference_cost(
    num_characters: int,
    num_objects: int,
    cost_per_image: float = DEFAULT_IMAGE_COST,
    character_candidates: int = CHARACTER_REF_CANDIDATES,
    object_candidates: int = OBJECT_REF_CANDIDATES
) -> tuple[float, float]:
    """
    Estimate cost for character and object reference sheet generation.

    Args:
        num_characters: Number of characters
        num_objects: Number of key objects
        cost_per_image: Cost per generated image
        character_candidates: Number of candidates per character
        object_candidates: Number of candidates per object

    Returns:
        Tuple of (character_ref_cost, object_ref_cost)
    """
    char_cost = num_characters * character_candidates * cost_per_image
    obj_cost = num_objects * object_candidates * cost_per_image
    return char_cost, obj_cost


def estimate_composition_cost(
    page_count: int,
    avg_panels_per_page: int = AVG_PANELS_PER_PAGE
) -> float:
    """
    Estimate cost for composition analysis (bubble placement, cropping).

    Args:
        page_count: Number of pages
        avg_panels_per_page: Average panels per page

    Returns:
        Composition analysis cost
    """
    total_panels = page_count * avg_panels_per_page
    tokens = total_panels * COMPOSITION_TOKENS_PER_PANEL
    # Uses flash model for composition analysis
    return (tokens / 1_000_000) * (GEMINI_FLASH_INPUT_PER_M + GEMINI_FLASH_OUTPUT_PER_M)


def estimate_total_cost(
    word_count: int,
    page_count: int,
    num_characters: int,
    num_objects: int,
    avg_panels_per_page: int = AVG_PANELS_PER_PAGE,
    margin_percent: float = DEFAULT_MARGIN_PERCENT,
    include_prescan: bool = True
) -> CostEstimate:
    """
    Calculate total estimated cost with margin fee.

    Args:
        word_count: Number of words in source document
        page_count: Target page count
        num_characters: Number of characters (from prescan or heuristic)
        num_objects: Number of key objects (from prescan or heuristic)
        avg_panels_per_page: Average panels per page
        margin_percent: Margin fee percentage (default 10%)
        include_prescan: Whether to include prescan cost

    Returns:
        CostEstimate with full breakdown
    """
    # Calculate component costs
    script_input, script_output, cache_savings = estimate_scripting_cost(
        word_count, page_count
    )

    image_cost = estimate_image_cost(page_count, avg_panels_per_page)
    char_ref_cost, obj_ref_cost = estimate_reference_cost(num_characters, num_objects)
    composition_cost = estimate_composition_cost(page_count, avg_panels_per_page)

    prescan_cost = PRESCAN_COST if include_prescan else 0.0

    # Build breakdown
    breakdown = CostBreakdown(
        prescan=prescan_cost,
        scripting_input=script_input,
        scripting_output=script_output,
        cache_savings=cache_savings,
        image_generation=image_cost,
        character_refs=char_ref_cost,
        object_refs=obj_ref_cost,
        composition_analysis=composition_cost,
    )

    # Calculate totals
    subtotal = (
        prescan_cost +
        script_input +
        script_output +
        image_cost +
        char_ref_cost +
        obj_ref_cost +
        composition_cost
    )

    margin_fee = subtotal * margin_percent
    total = subtotal + margin_fee

    total_panels = page_count * avg_panels_per_page

    return CostEstimate(
        word_count=word_count,
        pages=page_count,
        panels=total_panels,
        characters=num_characters,
        objects=num_objects,
        breakdown=breakdown,
        subtotal_vertex=subtotal,
        margin_fee=margin_fee,
        margin_percent=margin_percent,
        total_cost=total,
    )


def estimate_characters_from_word_count(word_count: int) -> int:
    """
    Heuristic to estimate number of characters from word count.

    Short stories: 2-4 characters
    Novellas: 4-6 characters
    Novels: 6-10 characters
    Epics: 10-15 characters

    Args:
        word_count: Number of words in document

    Returns:
        Estimated number of characters
    """
    if word_count < 20_000:
        return 3  # Short story
    elif word_count < 50_000:
        return 5  # Novella
    elif word_count < 120_000:
        return 8  # Novel
    else:
        return 12  # Epic


def estimate_objects_from_word_count(word_count: int) -> int:
    """
    Heuristic to estimate number of key objects from word count.

    Args:
        word_count: Number of words in document

    Returns:
        Estimated number of key objects
    """
    if word_count < 20_000:
        return 2  # Short story
    elif word_count < 50_000:
        return 4  # Novella
    elif word_count < 120_000:
        return 6  # Novel
    else:
        return 8  # Epic


def quick_estimate(
    word_count: int,
    page_count: Optional[int] = None,
    margin_percent: float = DEFAULT_MARGIN_PERCENT
) -> CostEstimate:
    """
    Quick cost estimate using heuristics (no prescan required).

    This provides a rough estimate before document analysis.
    For more accurate estimates, use estimate_total_cost() with
    actual character/object counts from prescan.

    Args:
        word_count: Number of words in document
        page_count: Optional page count (calculated from word count if not provided)
        margin_percent: Margin fee percentage

    Returns:
        CostEstimate with heuristic-based counts
    """
    from utils import calculate_page_count

    # Calculate page count if not provided
    if page_count is None:
        page_info = calculate_page_count(word_count)
        page_count = page_info['recommended']

    # Use heuristics for character/object counts
    num_characters = estimate_characters_from_word_count(word_count)
    num_objects = estimate_objects_from_word_count(word_count)

    return estimate_total_cost(
        word_count=word_count,
        page_count=page_count,
        num_characters=num_characters,
        num_objects=num_objects,
        margin_percent=margin_percent,
        include_prescan=False,  # Quick estimate doesn't include prescan
    )


def format_cost_summary(estimate: CostEstimate) -> str:
    """
    Format a human-readable cost summary.

    Args:
        estimate: CostEstimate object

    Returns:
        Formatted string summary
    """
    lines = [
        f"Estimated Cost Summary",
        f"=" * 40,
        f"Document: {estimate.word_count:,} words",
        f"Output: {estimate.pages} pages, ~{estimate.panels} panels",
        f"Characters: {estimate.characters}, Objects: {estimate.objects}",
        f"",
        f"Breakdown:",
        f"  Scripting (input):       ${estimate.breakdown.scripting_input:.4f}",
        f"  Scripting (output):      ${estimate.breakdown.scripting_output:.4f}",
        f"  Cache savings:          -${estimate.breakdown.cache_savings:.4f}",
        f"  Image generation:        ${estimate.breakdown.image_generation:.4f}",
        f"  Character references:    ${estimate.breakdown.character_refs:.4f}",
        f"  Object references:       ${estimate.breakdown.object_refs:.4f}",
        f"  Composition analysis:    ${estimate.breakdown.composition_analysis:.4f}",
        f"",
        f"Subtotal (Vertex AI):      ${estimate.subtotal_vertex:.4f}",
        f"Margin fee ({estimate.margin_percent*100:.0f}%):         ${estimate.margin_fee:.4f}",
        f"-" * 40,
        f"Total:                     ${estimate.total_cost:.2f}",
    ]
    return "\n".join(lines)
