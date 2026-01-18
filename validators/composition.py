"""
Image Composition Analysis module: LLM-based panel composition analysis.

Includes:
- ImageCompositionAnalyzer: Analyzes generated panels for subject regions,
  negative space, and safe crop bounds.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from PIL import Image

from google.genai import types
from config import config
from utils import get_client


# ==============================================================================
# IMAGE COMPOSITION ANALYZER - LLM-based panel composition analysis
# ==============================================================================

@dataclass
class CompositionAnalysisResult:
    """Result of analyzing a generated panel image for composition."""
    subject_regions: List[Dict] = field(default_factory=list)  # [{"type": "face", "bounds": {...}, "importance": 0.95}]
    negative_space_regions: List[Dict] = field(default_factory=list)  # [{"position": "top-left", "suitability": 0.9}]
    recommended_bubble_position: str = "top-left"
    bubble_confidence: float = 0.5
    safe_crop_bounds: Dict = field(default_factory=lambda: {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0})


class ImageCompositionAnalyzer:
    """
    Analyzes generated panel images to determine:
    - Subject regions (faces, important objects) to avoid cropping
    - Negative space suitable for text bubble placement
    - Safe crop bounds for smart cropping

    Uses LLM vision capabilities to analyze the image composition.
    """

    def __init__(self, output_base_dir: Path = None):
        self.output_base_dir = output_base_dir or Path("assets/output/pages")

    def get_analysis_path(self, page_num: int, panel_id: int) -> Path:
        """Returns the path for the composition analysis sidecar JSON."""
        return self.output_base_dir / f"page_{page_num}" / f"panel_{panel_id}_composition.json"

    def load_existing_analysis(self, page_num: int, panel_id: int) -> Optional[CompositionAnalysisResult]:
        """
        Load existing composition analysis from sidecar JSON if it exists.

        Args:
            page_num: Page number
            panel_id: Panel ID

        Returns:
            CompositionAnalysisResult if found, None otherwise
        """
        analysis_path = self.get_analysis_path(page_num, panel_id)
        if not analysis_path.exists():
            return None

        try:
            with open(analysis_path, "r") as f:
                data = json.load(f)

            return CompositionAnalysisResult(
                subject_regions=data.get("subject_regions", []),
                negative_space_regions=data.get("negative_space_regions", []),
                recommended_bubble_position=data.get("recommended_bubble_position", "top-left"),
                bubble_confidence=data.get("bubble_confidence", 0.5),
                safe_crop_bounds=data.get("safe_crop_bounds", {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0})
            )
        except Exception as e:
            print(f"   Failed to load composition analysis: {e}")
            return None

    def save_analysis(self, page_num: int, panel_id: int, result: CompositionAnalysisResult):
        """
        Save composition analysis to sidecar JSON.

        Args:
            page_num: Page number
            panel_id: Panel ID
            result: CompositionAnalysisResult to save
        """
        analysis_path = self.get_analysis_path(page_num, panel_id)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "subject_regions": result.subject_regions,
            "negative_space_regions": result.negative_space_regions,
            "recommended_bubble_position": result.recommended_bubble_position,
            "bubble_confidence": result.bubble_confidence,
            "safe_crop_bounds": result.safe_crop_bounds
        }

        with open(analysis_path, "w") as f:
            json.dump(data, f, indent=2)

    async def analyze_panel(self, image: Image.Image, panel_data: dict) -> CompositionAnalysisResult:
        """
        Analyze a generated panel image to detect subject regions and negative space.

        Args:
            image: PIL Image of the generated panel
            panel_data: Panel script data for context

        Returns:
            CompositionAnalysisResult with analysis data
        """
        characters = panel_data.get('characters', [])
        char_list = ", ".join(characters) if characters else "unspecified characters"
        visual_desc = panel_data.get('visual_description', '')[:200]

        analysis_prompt = f"""
You are an expert comic book compositor analyzing panel images for text bubble placement.

TASK: Analyze this panel image and identify:
1. SUBJECT REGIONS: Where are faces, important characters, or key action happening?
2. NEGATIVE SPACE: Which corners/edges have uncluttered areas suitable for text bubbles?
3. SAFE CROP BOUNDS: What area can be safely cropped without cutting off faces or key subjects?

PANEL CONTEXT:
- Characters: {char_list}
- Description: {visual_desc}

ANALYSIS REQUIREMENTS:
- Identify regions containing faces or important subjects (normalized 0.0-1.0 coordinates)
- Rate negative space regions by suitability for text (0.0-1.0)
- Recommend the best bubble position from: top-left, top-right, bottom-left, bottom-right
- Provide safe crop bounds that preserve all important subjects (especially faces)

Respond with ONLY a JSON object:
{{
    "subject_regions": [
        {{"type": "face", "bounds": {{"x_min": 0.3, "y_min": 0.1, "x_max": 0.7, "y_max": 0.5}}, "importance": 0.95}},
        {{"type": "action", "bounds": {{"x_min": 0.4, "y_min": 0.5, "x_max": 0.9, "y_max": 0.9}}, "importance": 0.7}}
    ],
    "negative_space_regions": [
        {{"position": "top-left", "suitability": 0.9}},
        {{"position": "top-right", "suitability": 0.3}},
        {{"position": "bottom-left", "suitability": 0.6}},
        {{"position": "bottom-right", "suitability": 0.2}}
    ],
    "recommended_bubble_position": "top-left",
    "bubble_confidence": 0.85,
    "safe_crop_bounds": {{"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 0.95}}
}}
"""

        try:
            response = await get_client().aio.models.generate_content(
                model=config.composition_analysis_model,
                contents=[analysis_prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "subject_regions": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "type": {"type": "STRING"},
                                        "bounds": {
                                            "type": "OBJECT",
                                            "properties": {
                                                "x_min": {"type": "NUMBER"},
                                                "y_min": {"type": "NUMBER"},
                                                "x_max": {"type": "NUMBER"},
                                                "y_max": {"type": "NUMBER"}
                                            }
                                        },
                                        "importance": {"type": "NUMBER"}
                                    }
                                }
                            },
                            "negative_space_regions": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "position": {"type": "STRING"},
                                        "suitability": {"type": "NUMBER"}
                                    }
                                }
                            },
                            "recommended_bubble_position": {"type": "STRING"},
                            "bubble_confidence": {"type": "NUMBER"},
                            "safe_crop_bounds": {
                                "type": "OBJECT",
                                "properties": {
                                    "x_min": {"type": "NUMBER"},
                                    "y_min": {"type": "NUMBER"},
                                    "x_max": {"type": "NUMBER"},
                                    "y_max": {"type": "NUMBER"}
                                }
                            }
                        },
                        "required": ["subject_regions", "negative_space_regions", "recommended_bubble_position", "bubble_confidence", "safe_crop_bounds"]
                    }
                )
            )

            result = response.parsed
            if result:
                return CompositionAnalysisResult(
                    subject_regions=result.get("subject_regions", []),
                    negative_space_regions=result.get("negative_space_regions", []),
                    recommended_bubble_position=result.get("recommended_bubble_position", "top-left"),
                    bubble_confidence=result.get("bubble_confidence", 0.5),
                    safe_crop_bounds=result.get("safe_crop_bounds", {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0})
                )

        except Exception as e:
            print(f"   Composition analysis failed: {e}")

        # Return default result on failure
        return CompositionAnalysisResult()
