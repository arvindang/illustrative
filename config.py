"""
Centralized configuration for Illustrative AI pipeline.
All model settings, rate limits, and paths are defined here.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded before PipelineConfig is instantiated
load_dotenv()


@dataclass
class PipelineConfig:
    """
    Single source of truth for all pipeline configuration.
    """

    # ==================== API Configuration ====================
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

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
    page_width: int = 2400
    page_height: int = 3200
    page_margin: int = 60
    panel_gutter: int = 40
    font_path: str = "fonts/PatrickHand-Regular.ttf"
    font_size: int = 60

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

        if not config.gemini_api_key:
            print("⚠️ Warning: GEMINI_API_KEY not set in environment")
            return False

        # Ensure output directories exist
        for directory in [config.output_dir, config.characters_dir, config.objects_dir,
                         config.pages_dir, config.final_pages_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        return True


# Global configuration instance
config = PipelineConfig()

# Validate on import
if __name__ == "__main__":
    if config.validate():
        print("✅ Configuration validated successfully")
    else:
        print("❌ Configuration validation failed")
