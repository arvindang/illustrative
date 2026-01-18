import os
import io
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from config import config
from typing import Optional, Dict, List, Tuple
from agents.layout_agent import LayoutAgent
from agents.export_agent import ExportAgent


class CompositorAgent:
    def __init__(
        self,
        script_path: str,
        base_output_dir: Path = None,
        layout_agent: LayoutAgent = None,
        export_agent: ExportAgent = None
    ):
        self.script_path = Path(script_path)

        if base_output_dir:
            self.base_dir = Path(base_output_dir)
            self.panels_dir = self.base_dir / "pages"
            self.output_dir = self.base_dir / "final_pages"
        else:
            self.panels_dir = config.pages_dir
            self.output_dir = config.final_pages_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration for page dimensions
        self.page_width = config.page_width
        self.page_height = config.page_height
        self.margin = config.page_margin
        self.gutter = config.panel_gutter

        # Initialize or use provided LayoutAgent
        if layout_agent:
            self.layout_agent = layout_agent
        else:
            self.layout_agent = LayoutAgent(
                page_width=self.page_width,
                page_height=self.page_height,
                margin=self.margin,
                gutter=self.gutter
            )

        # Initialize or use provided ExportAgent
        if export_agent:
            self.export_agent = export_agent
        else:
            self.export_agent = ExportAgent(output_dir=self.output_dir)

        # Load a font (Ensure you have a .ttf file in a 'fonts' folder)
        try:
            self.font = ImageFont.truetype(config.font_path, config.font_size)
        except:
            print("Warning: Font not found, using default.")
            self.font = ImageFont.load_default()

    def clean_dialogue(self, text: str) -> str:
        """
        Remove character name prefixes AND stage directions from dialogue.
        This prevents bubbles showing "Nemo: I shall not..." or "(SFX): CLANG!"

        Args:
            text: Raw dialogue text that may contain character name prefixes or stage directions

        Returns:
            Cleaned dialogue text without name prefixes or stage directions
        """
        if not text:
            return text

        cleaned = text.strip()

        # Remove patterns like "Name:" or "Name (aside):" at the start of dialogue
        # Matches: "Captain Nemo:", "Professor:", "Nemo (whispers):", etc.
        cleaned = re.sub(r'^[A-Z][a-zA-Z\s]+(\([^)]+\))?:\s*', '', cleaned)

        # Strip stage directions anywhere in text
        cleaned = re.sub(r'\(SFX\):\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(Internal Monologue\):\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(aside\):\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(thought bubble\):\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(whispers?\):\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(narration\):\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(voice[- ]?over\):\s*', '', cleaned, flags=re.IGNORECASE)

        # Remove leading/trailing quotes and whitespace
        return cleaned.strip().strip('"').strip()

    def wrap_text(self, text: str, max_width: int, font=None) -> str:
        """
        Wraps text to fit within a specified pixel width using actual font measurements.

        Args:
            text: The text to wrap
            max_width: Maximum width in pixels
            font: Font to use for measurement (defaults to self.font)

        Returns:
            Text with newlines inserted at appropriate positions
        """
        if not text:
            return ""

        if font is None:
            font = self.font

        words = text.split()
        if not words:
            return ""

        lines = []
        current_line = []

        # Create temporary image for measurement
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)

        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]

            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def _fit_text_to_bubble(self, text: str, max_width: int, max_height: int, padding: int = 30) -> tuple:
        """
        Fits text to bubble by reducing font size or truncating if needed.

        Args:
            text: The cleaned text to fit
            max_width: Maximum bubble width in pixels
            max_height: Maximum bubble height in pixels
            padding: Padding inside bubble

        Returns:
            Tuple of (wrapped_text, font_to_use, actual_font_size)
        """
        available_width = max_width - (padding * 2)
        available_height = max_height - (padding * 2)

        font_size = config.font_size
        min_font_size = 12  # Reduced from 16 to allow more text fitting

        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)

        while font_size >= min_font_size:
            try:
                test_font = ImageFont.truetype(config.font_path, font_size)
            except:
                test_font = self.font

            wrapped = self.wrap_text(text, available_width, font=test_font)

            # Measure height
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=test_font)
            height = bbox[3] - bbox[1]

            if height <= available_height:
                return wrapped, test_font, font_size

            font_size -= 4  # Reduce font size

        # Still too long - truncate with ellipsis at minimum font size
        try:
            final_font = ImageFont.truetype(config.font_path, min_font_size)
        except:
            final_font = self.font

        truncated = self._truncate_with_ellipsis(text, available_width, available_height, final_font)
        return truncated, final_font, min_font_size

    def _truncate_with_ellipsis(self, text: str, max_width: int, max_height: int, font) -> str:
        """
        Truncates text to fit within bounds, adding ellipsis.

        Args:
            text: Text to truncate
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            font: Font to use for measurement

        Returns:
            Truncated text with ellipsis if needed
        """
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)

        words = text.split()
        result_words = []

        for word in words:
            test_text = ' '.join(result_words + [word])
            wrapped = self.wrap_text(test_text + '...', max_width, font=font)

            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
            height = bbox[3] - bbox[1]

            if height > max_height:
                # Adding this word would overflow
                break

            result_words.append(word)

        if len(result_words) < len(words):
            # Text was truncated
            return self.wrap_text(' '.join(result_words) + '...', max_width, font=font)
        else:
            return self.wrap_text(text, max_width, font=font)

    def draw_caption_box(self, draw, text, panel_rect, position="top-left"):
        """Draws a rectangular caption box for narration/monologue with overflow handling."""
        panel_x, panel_y, panel_w, panel_h = panel_rect
        cleaned_text = self.clean_dialogue(text)

        if not cleaned_text:
            return

        padding = 20
        edge_margin = 30

        # Max bubble size: 45% of panel width, 30% of panel height (reduced to cover less panel)
        max_box_w = int(panel_w * 0.45)
        max_box_h = int(panel_h * 0.30)

        # Fit text with overflow handling
        wrapped_text, font_to_use, _ = self._fit_text_to_bubble(
            cleaned_text, max_box_w, max_box_h, padding=padding
        )

        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font_to_use)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        box_w = text_w + (padding * 2)
        box_h = text_h + (padding * 2)

        # Caption box positions (usually corners)
        if position == "top-right":
            x = panel_x + panel_w - box_w - edge_margin
            y = panel_y + edge_margin
        elif position == "bottom-left":
            x = panel_x + edge_margin
            y = panel_y + panel_h - box_h - edge_margin
        elif position == "bottom-right":
            x = panel_x + panel_w - box_w - edge_margin
            y = panel_y + panel_h - box_h - edge_margin
        else:  # top-left default
            x = panel_x + edge_margin
            y = panel_y + edge_margin

        # Draw Caption Box (Yellowish background, sharp corners)
        box_rect = [x, y, x + box_w, y + box_h]
        draw.rectangle(box_rect, fill="#FFFACD", outline="black", width=2)

        text_x = x + padding
        text_y = y + padding
        draw.multiline_text((text_x, text_y), wrapped_text, font=font_to_use, fill="black")

    def draw_speech_bubble(self, draw, text, panel_rect, position_code="top-left"):
        """Draws a rounded speech bubble with wrapped text and overflow handling."""
        # Check if this is actually a caption request via position code
        if position_code == "caption-box":
            self.draw_caption_box(draw, text, panel_rect)
            return

        panel_x, panel_y, panel_w, panel_h = panel_rect
        # Clean dialogue first (remove character name prefixes and stage directions)
        cleaned_text = self.clean_dialogue(text)

        if not cleaned_text:
            return

        padding = 30
        edge_margin = 40

        # Max bubble size: 40% of panel width, 30% of panel height (reduced to cover less panel)
        max_bubble_w = int(panel_w * 0.40)
        max_bubble_h = int(panel_h * 0.30)

        # Fit text with overflow handling (shrink font or truncate)
        wrapped_text, font_to_use, _ = self._fit_text_to_bubble(
            cleaned_text, max_bubble_w, max_bubble_h, padding=padding
        )

        # Calculate actual text size with the fitted font
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font_to_use)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        bubble_w = text_w + (padding * 2)
        bubble_h = text_h + (padding * 2)

        # Determine coordinates based on position_code
        if position_code == "top-right":
            x = panel_x + panel_w - bubble_w - edge_margin
            y = panel_y + edge_margin
        elif position_code == "bottom-left":
            x = panel_x + edge_margin
            y = panel_y + panel_h - bubble_h - edge_margin
        elif position_code == "bottom-right":
            x = panel_x + panel_w - bubble_w - edge_margin
            y = panel_y + panel_h - bubble_h - edge_margin
        else:  # top-left (default)
            x = panel_x + edge_margin
            y = panel_y + edge_margin

        # Draw bubble background
        bubble_rect = [x, y, x + bubble_w, y + bubble_h]
        draw.rounded_rectangle(bubble_rect, radius=15, fill="white", outline="black", width=3)

        # Draw text in bubble
        text_x = x + padding
        text_y = y + padding
        draw.multiline_text((text_x, text_y), wrapped_text, font=font_to_use, fill="black")

    def generate_all_layouts(self, script_data: List[Dict]) -> None:
        """Delegate to LayoutAgent for batch layout generation."""
        self.layout_agent.generate_all_layouts(script_data)

    def generate_smart_layout(self, panels: List[Dict], page_num: int = None) -> List[Tuple[int, int, int, int]]:
        """Delegate to LayoutAgent for smart layout generation."""
        return self.layout_agent.generate_smart_layout(panels, page_num)

    def calculate_layout(self, num_panels: int) -> List[Tuple[int, int, int, int]]:
        """Delegate to LayoutAgent for grid layout calculation."""
        return self.layout_agent.calculate_layout(num_panels)

    def load_composition_analysis(self, page_num: int, panel_id: int) -> Optional[Dict]:
        """
        Load composition analysis sidecar JSON for a panel if it exists.

        Args:
            page_num: Page number
            panel_id: Panel ID

        Returns:
            Dict with composition analysis data, or None if not found
        """
        analysis_path = self.panels_dir / f"page_{page_num}" / f"panel_{panel_id}_composition.json"

        if not analysis_path.exists():
            return None

        try:
            with open(analysis_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"   ⚠️ Failed to load composition analysis: {e}")
            return None

    def smart_crop(self, image: Image.Image, target_size: tuple, safe_crop_bounds: Optional[Dict] = None) -> Image.Image:
        """
        Crop image to target size while respecting safe crop bounds to avoid cutting off faces/subjects.

        Args:
            image: Source PIL Image
            target_size: Tuple of (width, height) for the target crop size
            safe_crop_bounds: Dict with normalized coordinates {"x_min", "y_min", "x_max", "y_max"}
                             If None, falls back to center crop

        Returns:
            Cropped and resized PIL Image
        """
        target_w, target_h = target_size
        src_w, src_h = image.size
        target_aspect = target_w / target_h
        src_aspect = src_w / src_h

        # If no safe bounds provided, fallback to center crop (ImageOps.fit behavior)
        if safe_crop_bounds is None:
            return ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS)

        # Convert normalized safe bounds to pixel coordinates
        x_min_safe = int(safe_crop_bounds.get("x_min", 0.0) * src_w)
        y_min_safe = int(safe_crop_bounds.get("y_min", 0.0) * src_h)
        x_max_safe = int(safe_crop_bounds.get("x_max", 1.0) * src_w)
        y_max_safe = int(safe_crop_bounds.get("y_max", 1.0) * src_h)

        # Ensure safe bounds are valid
        x_min_safe = max(0, x_min_safe)
        y_min_safe = max(0, y_min_safe)
        x_max_safe = min(src_w, x_max_safe)
        y_max_safe = min(src_h, y_max_safe)

        safe_w = x_max_safe - x_min_safe
        safe_h = y_max_safe - y_min_safe

        # Calculate crop dimensions that fit target aspect ratio
        if src_aspect > target_aspect:
            # Source is wider - crop width
            crop_h = src_h
            crop_w = int(src_h * target_aspect)
        else:
            # Source is taller - crop height
            crop_w = src_w
            crop_h = int(src_w / target_aspect)

        # Find crop position that maximizes overlap with safe region
        # while keeping the crop window within source bounds

        # Calculate center of safe region
        safe_center_x = (x_min_safe + x_max_safe) // 2
        safe_center_y = (y_min_safe + y_max_safe) // 2

        # Try to center crop on safe region center
        crop_x = max(0, min(src_w - crop_w, safe_center_x - crop_w // 2))
        crop_y = max(0, min(src_h - crop_h, safe_center_y - crop_h // 2))

        # Adjust if crop would cut into important regions
        # If safe region's top is above crop top, shift crop up
        if crop_y > y_min_safe:
            crop_y = max(0, y_min_safe)

        # If safe region's bottom is below crop bottom, shift crop down
        if crop_y + crop_h < y_max_safe:
            crop_y = min(src_h - crop_h, y_max_safe - crop_h)

        # Same for horizontal
        if crop_x > x_min_safe:
            crop_x = max(0, x_min_safe)

        if crop_x + crop_w < x_max_safe:
            crop_x = min(src_w - crop_w, x_max_safe - crop_w)

        # Perform crop and resize
        cropped = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        return cropped.resize(target_size, Image.Resampling.LANCZOS)

    def assemble_page(self, page_data):
        page_num = page_data['page_number']
        panels_list = page_data['panels']
        num_panels = len(panels_list)
        
        print(f"📄 Assembling Page {page_num} with {num_panels} panels...")
        
        # Create a blank white canvas
        canvas = Image.new("RGB", (self.page_width, self.page_height), "white")
        draw = ImageDraw.Draw(canvas)
        
        # Get dynamic layout coordinates (uses cache if available from batch generation)
        layout = self.generate_smart_layout(panels_list, page_num=page_num)
        
        # Fallback safety (if smart layout returned fewer items or failed silently to standard list)
        if len(layout) < num_panels:
             print("⚠️ Layout mismatch, using standard grid.")
             layout = self.calculate_layout(num_panels)
        
        for idx, panel in enumerate(panels_list):
            if idx >= len(layout):
                print(f"⚠️ Skipping extra panel {panel['panel_id']} (Layout maxed out)")
                continue

            # Get relative coordinates (x, y, w, h) from layout
            rel_x, rel_y, panel_w, panel_h = layout[idx]
            
            # Apply Absolute Page Margins
            pos_x = self.margin + rel_x
            pos_y = self.margin + rel_y
            
            panel_id = panel['panel_id']
            img_path = self.panels_dir / f"page_{page_num}" / f"panel_{panel_id}.png"
            
            if not img_path.exists():
                print(f"⚠️ Missing image for Page {page_num}, Panel {panel_id}")
                # Draw a placeholder rectangle if missing
                draw.rectangle([pos_x, pos_y, pos_x + panel_w, pos_y + panel_h], outline="red", width=5)
                continue
                
            # Load and resize the generated panel
            # Use smart_crop with composition analysis when available
            try:
                panel_img = Image.open(img_path)

                # Load composition analysis if available
                composition = None
                safe_crop_bounds = None
                if config.enable_image_composition_analysis:
                    composition = self.load_composition_analysis(page_num, panel_id)
                    if composition:
                        safe_crop_bounds = composition.get("safe_crop_bounds")

                # Use smart_crop with safe bounds, or fallback to center crop
                if safe_crop_bounds:
                    panel_img = self.smart_crop(panel_img, (panel_w, panel_h), safe_crop_bounds)
                else:
                    panel_img = ImageOps.fit(panel_img, (panel_w, panel_h), method=Image.Resampling.LANCZOS)

                # Paste the panel
                canvas.paste(panel_img, (pos_x, pos_y))

                panel_rect = (pos_x, pos_y, panel_w, panel_h)

                # 1. Draw Caption (if present)
                if panel.get('caption'):
                    self.draw_caption_box(draw, panel['caption'], panel_rect, position="top-left")

                # 2. Draw Dialogue bubble (if present)
                if panel.get('dialogue'):
                    # Start with script-defined bubble position
                    pos_code = panel.get('bubble_position', 'top-left')

                    # Override with image-derived position if confidence is high enough
                    if composition:
                        bubble_confidence = composition.get("bubble_confidence", 0.0)
                        recommended_pos = composition.get("recommended_bubble_position")
                        if bubble_confidence >= config.composition_analysis_confidence_threshold and recommended_pos:
                            print(f"   🎯 Using image-derived bubble position: {recommended_pos} (confidence: {bubble_confidence:.2f})")
                            pos_code = recommended_pos

                    # Avoid overlap: If caption exists at top-left, move dialogue to bottom row
                    if panel.get('caption'):
                        # Map top positions to bottom equivalents
                        top_to_bottom = {
                            "top-left": "bottom-right",
                            "top-right": "bottom-right",
                        }
                        pos_code = top_to_bottom.get(pos_code, pos_code)

                    self.draw_speech_bubble(draw, panel['dialogue'], panel_rect, pos_code)

            except Exception as e:
                print(f"❌ Error processing Panel {panel_id}: {e}")

        # Save the final page with PNG optimization
        output_path = self.output_dir / f"page_{page_num}.png"
        canvas.save(output_path, optimize=True)
        print(f"✅ Page {page_num} saved to {output_path}")

    def get_sorted_images(self) -> List[Path]:
        """Delegate to ExportAgent."""
        return self.export_agent.get_sorted_images()

    def export_pdf(self, output_path: Path) -> Optional[Path]:
        """Delegate to ExportAgent."""
        return self.export_agent.export_pdf(output_path)

    def export_epub(self, output_path: Path, title: str = "Graphic Novel", author: str = "Illustrate AI") -> Optional[Path]:
        """Delegate to ExportAgent."""
        return self.export_agent.export_epub(output_path, title=title, author=author)

    def export_and_upload(self, output_path: Path, novel_id: str, title: str = "Graphic Novel") -> Dict[str, Optional[str]]:
        """Delegate to ExportAgent."""
        return self.export_agent.export_and_upload(output_path, novel_id, title=title)

    def run(self):
        with open(self.script_path, "r") as f:
            script_data = json.load(f)

        # Batch-generate layouts for all pages in a single API call
        self.generate_all_layouts(script_data)

        for page in script_data:
            self.assemble_page(page)

        # Packaging
        print("📦 Packaging output...")
        
        if hasattr(self, 'base_dir'):
             output_base = self.base_dir / self.script_path.stem
        else:
             output_base = Path("assets/output") / self.script_path.stem
             
        self.export_pdf(output_base)
        self.export_epub(output_base, title=self.script_path.stem.replace("-", " ").title())

if __name__ == "__main__":
    script_path = "assets/output/20-thousand-leagues-under-the-sea_full_script.json"
    print(f"📖 Loading script: {script_path}")

    compositor = CompositorAgent(script_path)
    compositor.run()