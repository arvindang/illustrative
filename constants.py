"""
Configuration constants for Illustrative AI.
"""

ART_STYLES = [
    # Original
    "Lush Watercolor",
    "Gritty Noir",
    "Classic Comic Book",
    "Ukiyo-e Woodblock",
    "Cyberpunk Neon",
    "Botanical Illustration",
    # International Comic Traditions
    "Manga/Anime",
    "Ligne Claire (Franco-Belgian)",
    "Art Nouveau",
    "Traditional Indian Miniature",
    # Fine Art Movements
    "Impressionist",
    "Surrealist",
    "Expressionist",
    "Chiaroscuro",
    # Contemporary
    "Minimalist Line Art",
    "Vintage Pulp",
    "Sketch/Pencil Drawing",
]

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
