import os
import io
import json
import re
import textwrap
import uuid
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from google import genai
from google.genai import types
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from ebooklib import epub
from config import config
from utils import get_tpm_limiter, estimate_tokens_for_text, extract_token_usage

class CompositorAgent:
    def __init__(self, script_path: str, base_output_dir: Path = None):
        self.script_path = Path(script_path)
        
        if base_output_dir:
            self.base_dir = Path(base_output_dir)
            self.panels_dir = self.base_dir / "pages"
            self.output_dir = self.base_dir / "final_pages"
        else:
            self.panels_dir = config.pages_dir
            self.output_dir = config.final_pages_dir
            
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration for a standard 2x2 grid
        self.page_width = config.page_width
        self.page_height = config.page_height
        self.margin = config.page_margin
        self.gutter = config.panel_gutter

        # Cache for batched layouts (page_number -> layout list)
        self._layout_cache = {}

        # Load a font (Ensure you have a .ttf file in a 'fonts' folder)
        # Fallback to a basic system font if not found
        try:
            self.font = ImageFont.truetype(config.font_path, config.font_size)
        except:
            print("⚠️ Font not found, using default.")
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

    def generate_all_layouts(self, script_data: list):
        """
        Batch-generates layouts for ALL pages in a single API call.
        Stores results in self._layout_cache for use during assembly.
        """
        if not config.gemini_api_key:
            print("⚠️ No API key found. Using grid layouts for all pages.")
            return

        print(f"🧠 Batch-generating layouts for {len(script_data)} pages...")

        # Build context for all pages
        all_pages_context = []
        for page in script_data:
            page_num = page['page_number']
            panels = page['panels']
            panel_summaries = []
            for p in panels:
                desc = p.get('visual_description', '')[:100]
                panel_summaries.append(f"Panel {p['panel_id']}: {desc}")
            all_pages_context.append({
                "page_number": page_num,
                "panel_count": len(panels),
                "panel_summaries": panel_summaries
            })

        prompt = f"""
        Act as an expert Comic Book Layout Artist.
        Design layouts for {len(script_data)} pages of a graphic novel.

        PAGE SUMMARIES:
        {json.dumps(all_pages_context, indent=2)}

        REQUIREMENTS:
        1. For each page, output a layout with normalized coordinates (0.0 to 1.0).
        2. Emphasize important panels by making them larger.
        3. Ensure reading flow: Left->Right, Top->Bottom.
        4. Each page's panels must cover the full page (sum to 1.0 in both dimensions).

        OUTPUT FORMAT:
        JSON object with page_number as key:
        {{
          "1": [{{"panel_id": 1, "x": 0.0, "y": 0.0, "width": 1.0, "height": 0.5}}, ...],
          "2": [...],
          ...
        }}
        """

        # Track TPM for batch layout generation (sync context)
        batch_estimated = estimate_tokens_for_text(prompt)
        get_tpm_limiter().acquire_sync(batch_estimated)

        try:
            client = genai.Client(api_key=config.gemini_api_key)
            response = client.models.generate_content(
                model=config.layout_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            layouts_data = json.loads(response.text)

            # Update TPM with actual usage
            batch_input, batch_output = extract_token_usage(response)
            get_tpm_limiter().update_actual_usage(batch_estimated, batch_input + batch_output)

            # Process and cache layouts
            W = self.page_width - (2 * self.margin)
            H = self.page_height - (2 * self.margin)
            g = self.gutter

            for page_num_str, page_layout in layouts_data.items():
                page_num = int(page_num_str)
                final_layout = []

                for item in page_layout:
                    px_x = int(item['x'] * W)
                    px_y = int(item['y'] * H)
                    px_w = int(item['width'] * W)
                    px_h = int(item['height'] * H)

                    # Apply gutter
                    if px_x + px_w < W - 5:
                        px_w -= g
                    if px_y + px_h < H - 5:
                        px_h -= g

                    final_layout.append((px_x, px_y, px_w, px_h))

                self._layout_cache[page_num] = final_layout

            print(f"✅ Batched layouts generated for {len(self._layout_cache)} pages")

        except Exception as e:
            print(f"⚠️ Batch layout generation failed: {e}. Will use per-page fallback.")

    def generate_smart_layout(self, panels: list, page_num: int = None) -> list:
        """
        Uses Gemini to generate a dynamic, narrative-aware layout.
        Returns a list of tuples: [(rel_x, rel_y, w, h), ...] relative to the margin area.

        If page_num is provided and a cached layout exists, returns the cached version.
        """
        # Check cache first
        if page_num is not None and page_num in self._layout_cache:
            return self._layout_cache[page_num]

        if not config.gemini_api_key:
            print("⚠️ No API key found. Falling back to grid layout.")
            return self.calculate_layout(len(panels))

        print(f"🧠 Generating smart layout for {len(panels)} panels...")
        
        # Prepare context for the LLM
        panel_context = []
        for p in panels:
            desc = p.get('visual_description', '')
            diag = p.get('dialogue', '')
            panel_context.append(f"Panel {p['panel_id']}: {desc} | Dialogue length: {len(diag)}")
        
        context_str = "\n".join(panel_context)

        prompt = f"""
        Act as an expert Comic Book Layout Artist.
        Design a page layout for the following {len(panels)} panels.
        
        STORY CONTEXT:
        {context_str}

        REQUIREMENTS:
        1. Emphasize important panels (e.g., action, detailed settings) by making them larger.
        2. Ensure the reading flow is clear (Left->Right, Top->Bottom).
        3. Output a JSON list of bounding boxes.
        4. Coordinates must be normalized (0.0 to 1.0).
        5. No gaps between panels (except implicit gutter handled by renderer).
        6. Total width and height must sum to 1.0 (covering the page).

        OUTPUT FORMAT:
        JSON:
        [
          {{"panel_id": 1, "x": 0.0, "y": 0.0, "width": 1.0, "height": 0.3}},
          ...
        ]
        """

        # Track TPM for smart layout generation (sync context)
        smart_estimated = estimate_tokens_for_text(prompt)
        get_tpm_limiter().acquire_sync(smart_estimated)

        try:
            client = genai.Client(api_key=config.gemini_api_key)
            response = client.models.generate_content(
                model=config.layout_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "panel_id": {"type": "INTEGER"},
                                "x": {"type": "NUMBER"},
                                "y": {"type": "NUMBER"},
                                "width": {"type": "NUMBER"},
                                "height": {"type": "NUMBER"}
                            },
                            "required": ["panel_id", "x", "y", "width", "height"]
                        }
                    }
                )
            )
            
            layout_data = response.parsed

            # Update TPM with actual usage
            smart_input, smart_output = extract_token_usage(response)
            get_tpm_limiter().update_actual_usage(smart_estimated, smart_input + smart_output)

            # Convert normalized coordinates to pixel dimensions
            W = self.page_width - (2 * self.margin)
            H = self.page_height - (2 * self.margin)
            g = self.gutter
            
            final_layout = []
            # We need to sort by panel_id to ensure order matches the input list
            # Usually input is panel 1..N. Logic assumes input list is sorted by ID.
            layout_map = {item['panel_id']: item for item in layout_data}
            
            for p in panels:
                p_id = p['panel_id']
                if p_id not in layout_map:
                    print(f"⚠️ Layout missing panel {p_id}, falling back.")
                    return self.calculate_layout(len(panels))
                
                item = layout_map[p_id]
                
                # Scale
                px_x = int(item['x'] * W)
                px_y = int(item['y'] * H)
                px_w = int(item['width'] * W)
                px_h = int(item['height'] * H)
                
                # Apply Gutter (shrink from right/bottom)
                # This is a simple approximation. 
                # A robust system would calculate exact edges.
                # We'll just shrink width/height by gutter size unless it hits the edge.
                
                if px_x + px_w < W - 5: # If not touching right edge
                    px_w -= g
                if px_y + px_h < H - 5: # If not touching bottom edge
                    px_h -= g
                
                final_layout.append((px_x, px_y, px_w, px_h))
                
            return final_layout
            
        except Exception as e:
            print(f"⚠️ Smart Layout failed: {e}. Falling back to grid.")
            return self.calculate_layout(len(panels))

    def calculate_layout(self, num_panels):
        """
        Calculates panel coordinates (x, y, w, h) for 1-6 panels
        to fill the page, accounting for margins and gutters.
        Returns a list of tuples: [(x, y, w, h), ...] relative to the margin.
        """
        W = self.page_width - (2 * self.margin)
        H = self.page_height - (2 * self.margin)
        g = self.gutter
        
        layout = []
        
        if num_panels == 1:
            # 1 Panel: Full Page
            layout.append((0, 0, W, H))
            
        elif num_panels == 2:
            # 2 Panels: Stacked Vertical (Top / Bottom)
            h_half = (H - g) // 2
            layout.append((0, 0, W, h_half))
            layout.append((0, h_half + g, W, h_half))
            
        elif num_panels == 3:
            # 3 Panels: 2 Top (Half Width), 1 Bottom (Full Width)
            h_half = (H - g) // 2
            w_half = (W - g) // 2
            
            # Row 1 (2 panels)
            layout.append((0, 0, w_half, h_half))
            layout.append((w_half + g, 0, w_half, h_half))
            
            # Row 2 (1 panel, full width)
            layout.append((0, h_half + g, W, h_half))
            
        elif num_panels == 4:
            # 4 Panels: Standard 2x2 Grid
            h_half = (H - g) // 2
            w_half = (W - g) // 2
            
            # Row 1
            layout.append((0, 0, w_half, h_half))
            layout.append((w_half + g, 0, w_half, h_half))
            # Row 2
            layout.append((0, h_half + g, w_half, h_half))
            layout.append((w_half + g, h_half + g, w_half, h_half))
            
        elif num_panels == 5:
            # 5 Panels: 3 Rows (2 Top, 2 Middle, 1 Bottom)
            h_third = (H - 2 * g) // 3
            w_half = (W - g) // 2
            
            # Row 1
            layout.append((0, 0, w_half, h_third))
            layout.append((w_half + g, 0, w_half, h_third))
            
            # Row 2
            layout.append((0, h_third + g, w_half, h_third))
            layout.append((w_half + g, h_third + g, w_half, h_third))
            
            # Row 3 (Full Width)
            layout.append((0, (2 * h_third) + (2 * g), W, h_third))
            
        elif num_panels >= 6:
            # 6+ Panels: 3 Rows x 2 Cols
            # Limit to 6 for now, extra panels will be ignored or overlaid
            h_third = (H - 2 * g) // 3
            w_half = (W - g) // 2
            
            for row in range(3):
                y = row * (h_third + g)
                layout.append((0, y, w_half, h_third))
                layout.append((w_half + g, y, w_half, h_third))
                
            # If > 6, they won't get coordinates here, loop in assemble_page handles index check
            
        return layout

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
            # Use ImageOps.fit to maintain aspect ratio (center crop)
            try:
                panel_img = Image.open(img_path)
                panel_img = ImageOps.fit(panel_img, (panel_w, panel_h), method=Image.Resampling.LANCZOS)
                
                # Paste the panel
                canvas.paste(panel_img, (pos_x, pos_y))
                
                panel_rect = (pos_x, pos_y, panel_w, panel_h)

                # 1. Draw Caption (if present)
                if panel.get('caption'):
                    self.draw_caption_box(draw, panel['caption'], panel_rect, position="top-left")

                # 2. Draw Dialogue bubble (if present)
                if panel.get('dialogue'):
                    pos_code = panel.get('bubble_position', 'top-left')

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

    def get_sorted_images(self):
        """Returns a list of image paths sorted by page number."""
        images = list(self.output_dir.glob("page_*.png"))
        images.sort(key=lambda x: int(x.stem.split("_")[1]))
        return images

    def export_pdf(self, output_path: Path):
        """Bundles images into a PDF with JPEG compression for smaller file sizes."""
        images = self.get_sorted_images()
        if not images:
            print("⚠️ No images found to export to PDF.")
            return None

        pdf_path = output_path.with_suffix(".pdf")

        with Image.open(images[0]) as first_img:
            img_w, img_h = first_img.size

        c = pdf_canvas.Canvas(str(pdf_path), pagesize=(img_w, img_h))

        for img_path in images:
            # Convert to JPEG in memory for better compression
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                buffer = io.BytesIO()
                rgb_img.save(buffer, format="JPEG", quality=85, optimize=True)
                buffer.seek(0)
                c.drawImage(ImageReader(buffer), 0, 0, width=img_w, height=img_h)
            c.showPage()

        c.save()
        print(f"✅ PDF exported to {pdf_path}")
        return pdf_path

    def export_epub(self, output_path: Path, title="Graphic Novel", author="Illustrate AI"):
        """Bundles images into a Fixed-Layout EPUB 3 with JPEG compression."""
        images = self.get_sorted_images()
        if not images:
            print("⚠️ No images found to export to EPUB.")
            return None

        epub_path = output_path.with_suffix(".epub")
        book = epub.EpubBook()

        book.set_identifier(str(uuid.uuid4()))
        book.set_title(title)
        book.set_language('en')
        book.add_author(author)

        # Fixed Layout metadata (EPUB 3)
        book.add_metadata(None, 'meta', 'pre-paginated', {'property': 'rendition:layout'})
        book.add_metadata(None, 'meta', 'landscape', {'property': 'rendition:orientation'})
        book.add_metadata(None, 'meta', 'auto', {'property': 'rendition:spread'})

        spine = ['nav']
        manifest = []

        for i, img_path in enumerate(images):
            page_num = i + 1

            # Convert to JPEG for better compression
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                buffer = io.BytesIO()
                rgb_img.save(buffer, format="JPEG", quality=85, optimize=True)
                img_content = buffer.getvalue()

            epub_img = epub.EpubImage()
            epub_img.file_name = f'images/page_{page_num}.jpg'
            epub_img.media_type = 'image/jpeg'
            epub_img.content = img_content
            book.add_item(epub_img)

            html_content = f"""
            <?xml version="1.0" encoding="UTF-8"?>
            <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
                <head>
                    <title>Page {page_num}</title>
                    <meta name="viewport" content="width=1200, height=1600"/>
                    <style>
                        body {{ margin: 0; padding: 0; background-color: #FFFFFF; }}
                        img {{ width: 100%; height: 100%; display: block; }}
                    </style>
                </head>
                <body>
                    <img src="../images/page_{page_num}.jpg" alt="Page {page_num}"/>
                </body>
            </html>
            """

            item = epub.EpubHtml(title=f'Page {page_num}', file_name=f'text/page_{page_num}.xhtml', content=html_content)
            book.add_item(item)
            manifest.append(item)
            spine.append(item)

        book.toc = tuple(manifest)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = spine

        epub.write_epub(str(epub_path), book, {})
        print(f"✅ EPUB exported to {epub_path}")
        return epub_path

    def export_and_upload(self, output_path: Path, novel_id: str, title: str = "Graphic Novel") -> dict:
        """
        Export PDF and EPUB, then upload to Railway bucket.

        Args:
            output_path: Base path for export files
            novel_id: UUID of the novel record for storage key naming
            title: Title for the EPUB metadata

        Returns:
            Dict with pdf_storage_key and epub_storage_key
        """
        from storage.bucket import get_storage

        # Generate files locally
        pdf_path = self.export_pdf(output_path)
        epub_path = self.export_epub(output_path, title=title)

        result = {"pdf_storage_key": None, "epub_storage_key": None}

        # Upload to bucket if storage is configured
        storage = get_storage()
        if storage.is_configured():
            if pdf_path and pdf_path.exists():
                pdf_key = f"novels/{novel_id}/output.pdf"
                storage.upload_file(str(pdf_path), pdf_key, content_type="application/pdf")
                result["pdf_storage_key"] = pdf_key
                print(f"☁️ PDF uploaded to bucket: {pdf_key}")

            if epub_path and epub_path.exists():
                epub_key = f"novels/{novel_id}/output.epub"
                storage.upload_file(str(epub_path), epub_key, content_type="application/epub+zip")
                result["epub_storage_key"] = epub_key
                print(f"☁️ EPUB uploaded to bucket: {epub_key}")
        else:
            print("⚠️ Bucket not configured - files stored locally only")

        return result

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