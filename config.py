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


# ============================================================================
# SERVICE ACCOUNT KEY HANDLING (for Railway/production deployments)
# ============================================================================
# If GOOGLE_SERVICE_ACCOUNT_KEY env var contains JSON credentials,
# write them to a temp file and set GOOGLE_APPLICATION_CREDENTIALS.
# This allows Railway deployments to authenticate with Vertex AI.

def _setup_service_account_credentials():
    """
    Set up Google Application Default Credentials from env var if present.
    This is called at module load time to ensure credentials are available
    before any Google API clients are initialized.
    """
    import tempfile
    import atexit

    key_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY")
    if key_json and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            # Validate it's actually JSON
            import json
            json.loads(key_json)

            # Write to a temp file
            fd, path = tempfile.mkstemp(suffix='.json', prefix='gcp_credentials_')
            with os.fdopen(fd, 'w') as f:
                f.write(key_json)

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

            # Clean up on exit
            def cleanup():
                try:
                    os.unlink(path)
                except OSError:
                    pass
            atexit.register(cleanup)

        except json.JSONDecodeError:
            print("Warning: GOOGLE_SERVICE_ACCOUNT_KEY is not valid JSON")


_setup_service_account_credentials()


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

    # ==================== Access Control ====================
    # Comma-separated list of admin emails allowed to register/login
    # When set, only these emails can access the app
    admin_emails: str = field(default_factory=lambda: os.getenv("ADMIN_EMAILS", ""))

    def get_admin_emails(self) -> set:
        """Returns set of admin emails (lowercase). Empty set means open access."""
        if not self.admin_emails:
            return set()
        return {email.strip().lower() for email in self.admin_emails.split(",") if email.strip()}

    def is_admin_email(self, email: str) -> bool:
        """Check if email is an admin. Returns True if no admin restriction is set."""
        admin_set = self.get_admin_emails()
        if not admin_set:
            return True  # No restriction
        return email.strip().lower() in admin_set

    # ==================== Model Selection ====================
    # Text/Logic Models (Pipeline Pass Models)
    # gemini-3-pro-preview: Best reasoning (1M context, 65K output)
    # gemini-3-flash-preview: Fast + good reasoning (1M context, 65K output)
    scripting_model_global_context: str = "gemini-3-pro-preview"   # Pass 0-2: Complex analysis
    scripting_model_chapter_map: str = "gemini-3-flash-preview"    # Legacy
    scripting_model_page_script: str = "gemini-3-flash-preview"    # Pass 5: Scriptwriter (parallel)
    layout_model: str = "gemini-3-flash-preview"                   # Compositor

    # Additional Pass Models (can be tuned separately)
    adaptation_filter_model: str = "gemini-3-pro-preview"          # Pass 1.5: Adaptation Filter
    character_deep_dive_model: str = "gemini-3-pro-preview"        # Pass 3: Character Deep Dive
    asset_manifest_model: str = "gemini-3-pro-preview"             # Pass 4: Asset Manifest
    validation_model: str = "gemini-3-flash-preview"               # Pass 6: Validation

    # Image Models (priority order for fallback)
    # Gemini 3 Pro Image: 14 input images max, 32K output tokens, highest quality
    # Gemini 2.5 Flash Image: 3 input images max, GA fallback
    image_model_primary: str = "gemini-3-pro-image-preview"   # Best quality, 14 ref images
    image_model_fallback: str = "gemini-2.5-flash-image"      # GA fallback, 3 ref images max
    image_model_last_resort: str = "gemini-2.5-flash-image"   # Same as fallback

    # Character Design Models
    character_model_attributes: str = "gemini-3-flash-preview"  # Fast text/attributes
    character_model_image: str = "gemini-3-pro-image-preview"   # Best quality for refs

    # ==================== Rate Limiting ====================
    # Requests per minute (RPM) - these are defaults, adjusted by get_effective_rate_limits()
    # AI Studio defaults (conservative due to free tier limits)
    scripting_rpm: int = 5
    character_rpm: int = 5
    image_rpm: int = 5

    # Tokens per minute (TPM)
    tpm_limit: int = 1_000_000  # Google's default TPM limit
    tpm_safety_margin: float = 0.85  # Use 85% of limit for safety buffer
    tpm_enabled: bool = True  # Can disable for testing

    # Vertex AI rate limits (higher, adjust based on your actual quotas)
    # See VERTEX_AI_QUESTIONS.md for quota lookup instructions
    vertex_scripting_rpm: int = 30
    vertex_character_rpm: int = 10
    vertex_image_rpm: int = 10
    vertex_tpm_limit: int = 4_000_000

    def get_effective_rate_limits(self) -> dict:
        """
        Returns effective rate limits based on whether Vertex AI is enabled.
        Vertex AI typically has higher quotas than AI Studio's free tier.
        """
        if self.use_vertex_ai:
            return {
                "scripting_rpm": self.vertex_scripting_rpm,
                "character_rpm": self.vertex_character_rpm,
                "image_rpm": self.vertex_image_rpm,
                "tpm_limit": self.vertex_tpm_limit,
            }
        else:
            return {
                "scripting_rpm": self.scripting_rpm,
                "character_rpm": self.character_rpm,
                "image_rpm": self.image_rpm,
                "tpm_limit": self.tpm_limit,
            }

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

    # ==================== Retry & Resilience ====================
    max_page_retries: int = 2         # Max retries for failed page scripts
    retry_delay_base: float = 2.0     # Base delay for retries (exponential backoff)

    # ==================== Panel Layout Settings ====================
    # Panel size configuration (percentage of page area)
    panel_size_large_min_pct: int = 50   # Large panels take 50%+ of page
    panel_size_medium_min_pct: int = 25  # Medium panels take 25-50%
    panel_size_small_min_pct: int = 15   # Small panels take 15-25%

    # Default panel counts based on scene type
    default_panels_action: int = 5       # More panels for rapid action
    default_panels_dialogue: int = 4     # Medium panels for conversation
    default_panels_establishing: int = 3 # Fewer, larger panels for setting
    default_panels_climax: int = 3       # Dramatic panels for climactic moments

    # ==================== Spread & Cliffhanger Settings ====================
    max_spreads_per_50_pages: int = 3    # Don't overuse two-page spreads
    cliffhanger_pages_per_chapter: int = 3  # Aim for ~3 page-turn hooks per chapter

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

    # ==================== Quota/Fallback Behavior ====================
    # If True, stop pipeline when primary image model quota is exhausted
    # If False, fallback to secondary models (default behavior)
    stop_on_primary_quota_exhausted: bool = True

    # ==================== Image Composition Analysis ====================
    # LLM-based analysis of generated panels for smart cropping and bubble placement
    enable_image_composition_analysis: bool = True
    composition_analysis_model: str = "gemini-3-flash-preview"
    composition_analysis_confidence_threshold: float = 0.7  # Min confidence to override script bubble position
    reuse_existing_analysis: bool = True  # Reuse sidecar JSON on resume runs

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

            # Validate Vertex AI authentication
            auth_valid, auth_msg = validate_vertex_ai_auth(config.gcp_project)
            if not auth_valid:
                print(f"Warning: Vertex AI auth check failed: {auth_msg}")
                return False

            print(f"✓ Using Vertex AI (project: {config.gcp_project}, location: {config.gcp_location})")
            limits = config.get_effective_rate_limits()
            print(f"  Rate limits: {limits['scripting_rpm']} RPM (text), {limits['image_rpm']} RPM (image), {limits['tpm_limit']:,} TPM")
        elif not config.gemini_api_key:
            print("Warning: Neither GEMINI_API_KEY nor Vertex AI configured")
            return False
        else:
            print(f"✓ Using Gemini API key (AI Studio mode)")
            limits = config.get_effective_rate_limits()
            print(f"  Rate limits: {limits['scripting_rpm']} RPM (text), {limits['image_rpm']} RPM (image), {limits['tpm_limit']:,} TPM")

        # Ensure output directories exist
        for directory in [config.output_dir, config.characters_dir, config.objects_dir,
                         config.pages_dir, config.final_pages_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        return True


def validate_vertex_ai_auth(project_id: str = None) -> tuple:
    """
    Validate that Application Default Credentials (ADC) are configured for Vertex AI.

    Args:
        project_id: Optional GCP project ID to verify

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        import google.auth
        from google.auth import exceptions as auth_exceptions

        credentials, detected_project = google.auth.default()

        # Check if we have a project (either detected or provided)
        effective_project = project_id or detected_project
        if not effective_project:
            return False, "No GCP project detected. Set GOOGLE_CLOUD_PROJECT or run: gcloud config set project YOUR_PROJECT"

        # Try to get an access token to verify credentials are valid
        if hasattr(credentials, 'refresh'):
            try:
                import google.auth.transport.requests
                request = google.auth.transport.requests.Request()
                credentials.refresh(request)
            except auth_exceptions.RefreshError as e:
                return False, f"Credentials refresh failed: {e}. Run: gcloud auth application-default login"

        return True, f"ADC configured for project: {effective_project}"

    except ImportError:
        return False, "google-auth package not installed. Run: pip install google-auth"
    except Exception as e:
        error_msg = str(e)
        if "Could not automatically determine credentials" in error_msg:
            return False, "No Application Default Credentials found. Run: gcloud auth application-default login"
        return False, f"Auth error: {error_msg}"


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
