"""
Configuration constants for Illustrative AI.
"""

# Validated styles — tested for reference-based consistency
VALIDATED_STYLES = [
    "Lush Watercolor",
    "Classic Comic Book",
    "Ligne Claire (Franco-Belgian)",
    "Manga/Anime",
    "Gritty Noir",
    "Botanical Illustration",
]

# Experimental styles — may produce inconsistent results
EXPERIMENTAL_STYLES = [
    "Ukiyo-e Woodblock",
    "Cyberpunk Neon",
    "Art Nouveau",
    "Traditional Indian Miniature",
    "Vintage Pulp",
    "Sketch/Pencil Drawing",
    "Minimalist Line Art",
    "Chiaroscuro",
]

# All available styles (backward compatible)
ART_STYLES = VALIDATED_STYLES + EXPERIMENTAL_STYLES

# Style-specific prompt fragments for better consistency
STYLE_PROMPT_FRAGMENTS = {
    "Lush Watercolor": "soft color bleeds, visible paper texture, ethereal lighting, dreamlike quality",
    "Classic Comic Book": "bold outlines, flat colors, Ben-Day dots, dynamic poses, classic 4-color printing aesthetic",
    "Ligne Claire (Franco-Belgian)": "uniform line weight, flat colors, clean precise linework, Herge/Moebius influence",
    "Manga/Anime": "manga panel conventions, clean black linework, dynamic angles, expressive eyes, speed lines for motion",
    "Gritty Noir": "heavy shadows, high contrast, limited palette, chiaroscuro lighting, Frank Miller influence",
    "Botanical Illustration": "precise detail, scientific accuracy, natural color palette, fine line work, naturalist style",
    "Ukiyo-e Woodblock": "flat areas of color, bold outlines, woodblock print texture, Japanese composition",
    "Cyberpunk Neon": "neon lighting, dark backgrounds, holographic effects, rain-slicked surfaces",
    "Art Nouveau": "flowing organic lines, decorative borders, natural motifs, Mucha-inspired",
    "Traditional Indian Miniature": "flat perspective, rich colors, gold detailing, intricate patterns",
    "Vintage Pulp": "dramatic lighting, saturated colors, action-oriented composition, retro aesthetic",
    "Sketch/Pencil Drawing": "graphite texture, cross-hatching, visible pencil strokes, tonal gradation",
    "Minimalist Line Art": "single weight lines, minimal detail, negative space, clean composition",
    "Chiaroscuro": "dramatic light-dark contrast, Caravaggio-inspired, volumetric lighting",
}

NARRATIVE_TONES = [
    # Original
    "Heroic",
    "Suspenseful",
    "Melancholic",
    "Whimsical",
    "Dark Fantasy",
    "Educational",
    # Literary & Sophisticated
    "Philosophical",
    "Satirical",
    "Romantic",
    "Contemplative",
    "Tragic",
    # Genre Hybrids
    "Noir Detective",
    "Cosmic Horror",
    "Gothic",
]

# Page count limits
MIN_PAGES_PRODUCTION = 10
MAX_PAGES_PRODUCTION = 200
RECOMMENDED_MAX_PAGES = 150  # Quality/time sweet spot

MIN_PAGES_TEST = 1
DEFAULT_PAGES_TEST = 3

# Word density thresholds (words per page)
DENSITY_SHORT_STORY = 500    # < 20k words
DENSITY_NOVELLA = 750        # 20k-50k words
DENSITY_NOVEL = 1000         # 50k-120k words
DENSITY_EPIC = 1200          # > 120k words

# Word count category thresholds
BRACKET_SHORT = 20_000
BRACKET_NOVELLA = 50_000
BRACKET_NOVEL = 120_000
