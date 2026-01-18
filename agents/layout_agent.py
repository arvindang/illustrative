"""
Layout Agent: LLM-powered page layout generation.

Handles:
- Batch layout generation for all pages
- Smart narrative-aware layouts
- Grid fallback layouts

This agent is batch-capable for efficient processing.
"""

import json
from typing import Dict, List, Tuple, Optional

from google.genai import types
from config import config
from utils import get_tpm_limiter, estimate_tokens_for_text, extract_token_usage, get_client


class LayoutAgent:
    """
    Generates page layouts using LLM or grid fallback.

    This agent supports batch processing - layouts for all pages
    can be generated in a single API call.
    """

    def __init__(
        self,
        page_width: int = None,
        page_height: int = None,
        margin: int = None,
        gutter: int = None
    ):
        """
        Initialize the LayoutAgent.

        Args:
            page_width: Page width in pixels (default from config)
            page_height: Page height in pixels (default from config)
            margin: Page margin in pixels (default from config)
            gutter: Gutter between panels in pixels (default from config)
        """
        self.page_width = page_width or config.page_width
        self.page_height = page_height or config.page_height
        self.margin = margin or config.page_margin
        self.gutter = gutter or config.panel_gutter

        # Cache for batched layouts (page_number -> layout list)
        self._layout_cache: Dict[int, List[Tuple[int, int, int, int]]] = {}

    async def run(self, script_data: List[Dict]) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Generate layouts for all pages (async version).

        Args:
            script_data: List of page dictionaries from script

        Returns:
            Dict mapping page_number to list of (x, y, w, h) tuples
        """
        self.generate_all_layouts(script_data)
        return self._layout_cache

    def generate_all_layouts(self, script_data: List[Dict]) -> None:
        """
        Batch-generates layouts for ALL pages in a single API call.
        Stores results in self._layout_cache for use during assembly.

        Args:
            script_data: List of page dictionaries from script
        """
        if not config.gemini_api_key:
            print("Warning: No API key found. Using grid layouts for all pages.")
            return

        print(f"Batch-generating layouts for {len(script_data)} pages...")

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
            response = get_client().models.generate_content(
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

            print(f"Batched layouts generated for {len(self._layout_cache)} pages")

        except Exception as e:
            print(f"Warning: Batch layout generation failed: {e}. Will use per-page fallback.")

    def generate_smart_layout(self, panels: List[Dict], page_num: int = None) -> List[Tuple[int, int, int, int]]:
        """
        Uses Gemini to generate a dynamic, narrative-aware layout.
        Returns a list of tuples: [(rel_x, rel_y, w, h), ...] relative to the margin area.

        If page_num is provided and a cached layout exists, returns the cached version.

        Args:
            panels: List of panel dictionaries
            page_num: Optional page number for cache lookup

        Returns:
            List of (x, y, width, height) tuples in pixels
        """
        # Check cache first
        if page_num is not None and page_num in self._layout_cache:
            return self._layout_cache[page_num]

        if not config.gemini_api_key:
            print("Warning: No API key found. Falling back to grid layout.")
            return self.calculate_layout(len(panels))

        print(f"Generating smart layout for {len(panels)} panels...")

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
            response = get_client().models.generate_content(
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
            layout_map = {item['panel_id']: item for item in layout_data}

            for p in panels:
                p_id = p['panel_id']
                if p_id not in layout_map:
                    print(f"Warning: Layout missing panel {p_id}, falling back.")
                    return self.calculate_layout(len(panels))

                item = layout_map[p_id]

                # Scale
                px_x = int(item['x'] * W)
                px_y = int(item['y'] * H)
                px_w = int(item['width'] * W)
                px_h = int(item['height'] * H)

                # Apply Gutter (shrink from right/bottom)
                if px_x + px_w < W - 5:  # If not touching right edge
                    px_w -= g
                if px_y + px_h < H - 5:  # If not touching bottom edge
                    px_h -= g

                final_layout.append((px_x, px_y, px_w, px_h))

            return final_layout

        except Exception as e:
            print(f"Warning: Smart Layout failed: {e}. Falling back to grid.")
            return self.calculate_layout(len(panels))

    def calculate_layout(self, num_panels: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculates panel coordinates (x, y, w, h) for 1-6 panels
        to fill the page, accounting for margins and gutters.
        Returns a list of tuples: [(x, y, w, h), ...] relative to the margin.

        Args:
            num_panels: Number of panels to layout

        Returns:
            List of (x, y, width, height) tuples in pixels
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
            h_third = (H - 2 * g) // 3
            w_half = (W - g) // 2

            for row in range(3):
                y = row * (h_third + g)
                layout.append((0, y, w_half, h_third))
                layout.append((w_half + g, y, w_half, h_third))

        return layout

    def get_layout(self, page_num: int, panels: List[Dict]) -> List[Tuple[int, int, int, int]]:
        """
        Get layout for a page, using cache or generating on-demand.

        Args:
            page_num: Page number
            panels: List of panel dictionaries

        Returns:
            List of (x, y, width, height) tuples in pixels
        """
        if page_num in self._layout_cache:
            return self._layout_cache[page_num]

        # Fall back to grid layout if not in cache
        return self.calculate_layout(len(panels))

    def clear_cache(self):
        """Clear the layout cache."""
        self._layout_cache.clear()


if __name__ == "__main__":
    # Example usage
    agent = LayoutAgent()

    # Test grid layouts
    for n in range(1, 7):
        layout = agent.calculate_layout(n)
        print(f"Layout for {n} panels: {layout}")
