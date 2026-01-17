"""
Centralized configuration for Illustrative AI pipeline.
All model settings, rate limits, and paths are defined here.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded before PipelineConfig is instantiated
load_dotenv()


@dataclass
class PipelineConfig:
    """
    Single source of truth for all pipeline configuration.
    """

    # ==================== API Configuration ====================
    # Legacy Gemini API key (for backwards compatibility)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # Vertex AI settings (recommended for production - no daily limits)
    gcp_project: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    gcp_location: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
    use_vertex_ai: bool = field(default_factory=lambda: os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true")

    # ==================== Database Configuration ====================
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    jwt_secret_key: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", "dev-secret-change-me"))
    encryption_key: str = field(default_factory=lambda: os.getenv("ENCRYPTION_KEY", ""))

    # ==================== Storage Configuration ====================
    bucket_name: str = field(default_factory=lambda: os.getenv("BUCKET", ""))
    bucket_endpoint: str = field(default_factory=lambda: os.getenv("BUCKET_ENDPOINT", "https://storage.railway.app"))
    bucket_access_key: str = field(default_factory=lambda: os.getenv("BUCKET_ACCESS_KEY_ID", ""))
    bucket_secret_key: str = field(default_factory=lambda: os.getenv("BUCKET_SECRET_ACCESS_KEY", ""))
    bucket_region: str = field(default_factory=lambda: os.getenv("BUCKET_REGION", "auto"))

    # ==================== API URL (for Streamlit to call FastAPI) ====================
    api_url: str = field(default_factory=lambda: os.getenv("API_URL", "http://localhost:8000"))

    # ==================== Model Selection ====================
    # Text/Logic Models
    scripting_model_global_context: str = "gemini-2.5-flash"
    scripting_model_chapter_map: str = "gemini-2.5-flash"
    scripting_model_page_script: str = "gemini-2.5-flash"
    layout_model: str = "gemini-2.5-flash"

    # Image Models (priority order for fallback)
    image_model_primary: str = "nano-banana-pro-preview"
    image_model_fallback: str = "gemini-3-pro-image-preview"
    image_model_last_resort: str = "gemini-2.5-flash-image"

    # Character Design Models
    character_model_attributes: str = "gemini-3-flash-preview"
    character_model_image: str = "gemini-3-pro-image-preview"

    # ==================== Rate Limiting ====================
    # Requests per minute (RPM)
    scripting_rpm: int = 5
    character_rpm: int = 5
    image_rpm: int = 5

    # Tokens per minute (TPM)
    tpm_limit: int = 1_000_000  # Google's default TPM limit
    tpm_safety_margin: float = 0.85  # Use 85% of limit for safety buffer
    tpm_enabled: bool = True  # Can disable for testing

    # ==================== Page Allocation ====================
    # Word density thresholds (words per page)
    density_short_story: int = 500    # < 20k words
    density_novella: int = 750        # 20k-50k words
    density_novel: int = 1000         # 50k-120k words
    density_epic: int = 1200          # > 120k words

    # Page count limits
    min_pages_production: int = 10
    max_pages_production: int = 200
    recommended_max_pages: int = 150  # Quality/time sweet spot
    default_pages_test: int = 3

    # ==================== File Paths ====================
    output_dir: Path = Path("assets/output")
    input_dir: Path = Path("assets/input")
    characters_dir: Path = Path("assets/output/characters")
    objects_dir: Path = Path("assets/output/objects")
    pages_dir: Path = Path("assets/output/pages")
    final_pages_dir: Path = Path("assets/output/final_pages")

    # ==================== Composition Settings ====================
    page_width: int = 1200
    page_height: int = 1600
    page_margin: int = 30
    panel_gutter: int = 20
    font_path: str = "fonts/PatrickHand-Regular.ttf"
    font_size: int = 24  # Reduced from 32 to cover less panel area

    # ==================== Image Settings ====================
    image_aspect_ratio: str = "4:3"
    reference_image_aspect_ratio: str = "1:1"

    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration settings.
        Returns True if configuration is valid, False otherwise.
        """
        config = cls()

        # Check for valid API configuration (either Vertex AI or Gemini API key)
        if config.use_vertex_ai:
            if not config.gcp_project:
                print("Warning: GOOGLE_CLOUD_PROJECT not set but GOOGLE_GENAI_USE_VERTEXAI=true")
                return False
            print(f"✓ Using Vertex AI (project: {config.gcp_project}, location: {config.gcp_location})")
        elif not config.gemini_api_key:
            print("Warning: Neither GEMINI_API_KEY nor Vertex AI configured")
            return False

        # Ensure output directories exist
        for directory in [config.output_dir, config.characters_dir, config.objects_dir,
                         config.pages_dir, config.final_pages_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        return True


# ==================== ERA CONSTRAINT TEMPLATES ====================
# Pre-built constraints for common historical periods to ensure period accuracy
ERA_CONSTRAINTS = {
    "1860s Victorian / Nautical": """
Setting: 1860s Victorian era, maritime/underwater exploration.
SHIPS: Only sailing ships, wooden merchant vessels, early coal-powered steamships with paddle wheels or single screws. NO modern vessels.
CLOTHING (Men): Frock coats, waistcoats, cravats, top hats or bowler hats, heavy wool overcoats. Sailors wear traditional 19th-century naval uniforms.
CLOTHING (Women): Crinolines, full skirts, high collars, bonnets.
TECHNOLOGY: Gas lamps, telegraphs, early photography, mechanical instruments. NO electric lights, telephones, or modern electronics.
DIVING EQUIPMENT: Brass and copper diving helmets, canvas/rubber diving suits with riveted metal plates, air hoses connected to surface pumps. NO SCUBA gear.
WEAPONS: Harpoons, single-shot rifles, pistols, swords. NO automatic weapons.
INTERIORS: Victorian ornate style - mahogany wood, brass fittings, velvet upholstery, oil paintings, gas lighting.
FORBIDDEN: Automobiles, aircraft, plastic, electric lights (unless specifically from the Nautilus), modern clothing.
""",
    "1880s American West": """
Setting: 1880s American frontier.
CLOTHING: Cowboys wear wide-brimmed hats, bandanas, leather vests, chaps, spurs. Townspeople wear Victorian-era clothing.
TRANSPORTATION: Horses, stagecoaches, steam locomotives. NO automobiles.
WEAPONS: Single-action revolvers (Colt Peacemaker), lever-action rifles (Winchester), shotguns. NO automatic weapons.
BUILDINGS: Wooden frontier buildings, saloons with swinging doors, general stores, livery stables.
TECHNOLOGY: Telegraph, oil lamps, basic steam power. NO electricity, telephones.
FORBIDDEN: Modern vehicles, electric lights, plastic, automatic weapons, modern clothing.
""",
    "Medieval European": """
Setting: Medieval Europe (roughly 1100-1400 AD).
CLOTHING: Tunics, cloaks, leather boots, chainmail for warriors. Nobility wears velvet, silk, fur-trimmed garments, circlets.
WEAPONS: Swords, maces, axes, longbows, crossbows, halberds. NO firearms.
ARMOR: Chainmail, plate armor (later period), leather armor, shields with heraldic designs.
TRANSPORTATION: Horses, ox-carts, sailing ships. NO wheeled carriages (common only later).
BUILDINGS: Stone castles, thatched cottages, Gothic cathedrals, timber-framed buildings.
TECHNOLOGY: Forges, water mills, basic mechanical devices. NO clocks (except large tower clocks late period).
FORBIDDEN: Firearms, printed books (before 1450), glass windows in common buildings, potatoes/tomatoes (New World crops).
""",
    "Ancient Rome": """
Setting: Roman Empire (roughly 100 BC - 400 AD).
CLOTHING: Togas for citizens, tunics for common people, military wear includes lorica segmentata, sandals. Women wear stolas and pallas.
WEAPONS: Gladius (short sword), pilum (javelin), scutum (shield), bows.
ARMOR: Lorica segmentata, chainmail (lorica hamata), bronze helmets, leather armor.
TRANSPORTATION: Chariots, horses, litters (for wealthy), Roman roads, galleys and triremes.
BUILDINGS: Columns, arches, aqueducts, amphitheaters, insulae (apartment blocks), villas with atriums.
TECHNOLOGY: Roman concrete, aqueducts, hypocaust heating, oil lamps.
FORBIDDEN: Stirrups (not invented yet), modern materials, glass windows (rare), paper (use scrolls).
""",
    "1920s Art Deco": """
Setting: 1920s Roaring Twenties, Jazz Age.
CLOTHING (Men): Three-piece suits, fedoras, wing-tip shoes, bow ties.
CLOTHING (Women): Flapper dresses, cloche hats, long pearl necklaces, bobbed hair.
TRANSPORTATION: Early automobiles (Model T, luxury cars), steam trains, ocean liners, early biplanes.
ARCHITECTURE: Art Deco style - geometric patterns, chrome, glass, zigzag motifs, skyscrapers.
TECHNOLOGY: Telephones (candlestick style), radios, early film cameras, electric lights.
ENTERTAINMENT: Jazz clubs, speakeasies, silent films transitioning to talkies.
FORBIDDEN: Television, plastic, modern cars, casual clothing like jeans/t-shirts.
""",
    "Feudal Japan": """
Setting: Feudal Japan (Edo period, 1603-1868).
CLOTHING: Kimono, hakama, obi. Samurai wear distinctive topknots (chonmage). Armor is yoroi with kabuto helmets.
WEAPONS: Katana, wakizashi, tanto, yumi (bow), yari (spear), naginata.
ARCHITECTURE: Wooden buildings with sliding shoji doors, tatami mats, curved roofs, castle towns.
TRANSPORTATION: Horses, palanquins (kago), boats. NO wheeled transport common.
TECHNOLOGY: Swords, traditional crafts, early firearms (tanegashima). NO electricity.
CULTURE: Tea ceremony, kabuki, geisha, strict social hierarchy.
FORBIDDEN: Western clothing, modern technology, Christianity symbols (persecuted in Edo period).
""",
    "Custom (Enter Below)": ""
}


# Global configuration instance
config = PipelineConfig()

# Validate on import
if __name__ == "__main__":
    if config.validate():
        print("✅ Configuration validated successfully")
    else:
        print("❌ Configuration validation failed")
