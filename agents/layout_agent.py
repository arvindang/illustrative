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

        Now includes story-driven layout signals from the scripting pipeline:
        - recommended_splash: One panel should dominate (60%+ of page)
        - is_full_bleed: Single dramatic full-page image
        - is_cliffhanger: Final panel should be emphasized
        - panel_size: Per-panel size hints (large/medium/small)
        - shot_type: Cinematic framing context

        Args:
            script_data: List of page dictionaries from script
        """
        if not config.gemini_api_key:
            print("Warning: No API key found. Using story-aware grid layouts for all pages.")
            # Use story-aware fallback instead of returning early
            for page in script_data:
                page_num = page['page_number']
                panels = page['panels']
                panel_hints = [
                    {
                        'panel_size': p.get('panel_size', 'medium'),
                        'shot_type': p.get('shot_type', 'medium')
                    }
                    for p in panels
                ]
                self._layout_cache[page_num] = self.calculate_story_aware_layout(
                    num_panels=len(panels),
                    panel_hints=panel_hints,
                    recommended_splash=page.get('recommended_splash', False),
                    is_full_bleed=page.get('is_full_bleed', False),
                    is_cliffhanger=page.get('is_cliffhanger', False)
                )
            return

        print(f"Batch-generating story-aware layouts for {len(script_data)} pages...")

        # Build enriched context for all pages with story signals
        all_pages_context = []
        for page in script_data:
            page_num = page['page_number']
            panels = page['panels']

            # Build detailed panel info with layout hints
            panel_details = []
            for p in panels:
                panel_details.append({
                    "panel_id": p['panel_id'],
                    "panel_size": p.get('panel_size', 'medium'),  # large/medium/small
                    "shot_type": p.get('shot_type', 'medium'),    # establishing/close-up/etc
                    "description": p.get('visual_description', '')[:120]
                })

            all_pages_context.append({
                "page_number": page_num,
                "panel_count": len(panels),
                "scene_type": page.get('scene_type', 'dialogue'),
                # Story-driven layout signals from blueprint
                "recommended_splash": page.get('recommended_splash', False),
                "is_full_bleed": page.get('is_full_bleed', False),
                "is_cliffhanger": page.get('is_cliffhanger', False),
                "panels": panel_details
            })

        prompt = f"""
        Act as an expert Comic Book Layout Artist for a DIGITAL graphic novel (readers see one page at a time).
        Design layouts for {len(script_data)} pages.

        PAGE DATA (includes story-driven layout signals):
        {json.dumps(all_pages_context, indent=2)}

        STORY-DRIVEN LAYOUT RULES (CRITICAL):
        1. **recommended_splash=true**: ONE panel MUST occupy 60%+ of the page area. This is a dramatic moment.
        2. **is_full_bleed=true**: Create a SINGLE full-page panel (1 panel covering entire page). Epic/climactic moment.
        3. **is_cliffhanger=true**: The FINAL panel should be larger/emphasized (bottom-right position, dramatic sizing).
        4. **panel_size hints**: Use these to determine relative sizes:
           - "large": 50%+ of page area (dramatic reveals, action climaxes, establishing shots)
           - "medium": 25-50% of page area (standard storytelling)
           - "small": 15-25% of page area (rapid action, reactions, transitions)
        5. **shot_type context**:
           - "establishing"/"wide": Often larger panels to show environment
           - "close-up"/"extreme-close-up": Can be smaller but impactful
           - "birds-eye"/"worms-eye": Often benefit from larger sizing for dramatic effect

        GENERAL LAYOUT PRINCIPLES:
        - Ensure reading flow: Left→Right, Top→Bottom (manga-style right-to-left is NOT used)
        - Vary panel shapes for visual interest (not all rectangles the same aspect ratio)
        - Coordinates are normalized (0.0 to 1.0), will be scaled to pixel dimensions
        - Each page's panels must tile the full page (no gaps, no overlaps)

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

    def calculate_story_aware_layout(
        self,
        num_panels: int,
        panel_hints: List[Dict] = None,
        recommended_splash: bool = False,
        is_full_bleed: bool = False,
        is_cliffhanger: bool = False
    ) -> List[Tuple[int, int, int, int]]:
        """
        Story-aware fallback layout that uses panel_size hints and page-level signals.

        Unlike the basic calculate_layout(), this method:
        - Respects panel_size (large/medium/small) hints from the scriptwriter
        - Handles recommended_splash (one dominant 60%+ panel)
        - Handles is_full_bleed (single full-page panel)
        - Handles is_cliffhanger (emphasize final panel)

        Args:
            num_panels: Number of panels
            panel_hints: List of dicts with 'panel_size' and 'shot_type' per panel
            recommended_splash: If True, one panel should dominate (60%+)
            is_full_bleed: If True, create single full-page layout
            is_cliffhanger: If True, emphasize the final panel

        Returns:
            List of (x, y, width, height) tuples in pixels
        """
        W = self.page_width - (2 * self.margin)
        H = self.page_height - (2 * self.margin)
        g = self.gutter

        # Full bleed: single panel covering entire page
        if is_full_bleed or num_panels == 1:
            return [(0, 0, W, H)]

        # No hints? Fall back to standard grid
        if not panel_hints:
            return self.calculate_layout(num_panels)

        # Find the "large" panel index (splash candidate)
        large_idx = None
        for i, hint in enumerate(panel_hints):
            if hint.get('panel_size') == 'large':
                large_idx = i
                break

        # If recommended_splash but no explicit large panel, make the first panel large
        if recommended_splash and large_idx is None:
            large_idx = 0

        # If cliffhanger and no large panel marked, make the LAST panel larger
        if is_cliffhanger and large_idx is None:
            large_idx = num_panels - 1

        # Splash layout: one dominant panel + smaller supporting panels
        if large_idx is not None and num_panels >= 2:
            return self._splash_layout(num_panels, large_idx, W, H, g)

        # Count panel sizes to determine layout strategy
        size_counts = {'large': 0, 'medium': 0, 'small': 0}
        for hint in panel_hints:
            size = hint.get('panel_size', 'medium')
            size_counts[size] = size_counts.get(size, 0) + 1

        # If mostly small panels (rapid action), use tighter grid
        if size_counts['small'] >= num_panels * 0.6:
            return self._action_grid_layout(num_panels, W, H, g)

        # Default: use standard grid layout
        return self.calculate_layout(num_panels)

    def _splash_layout(
        self,
        num_panels: int,
        splash_idx: int,
        W: int,
        H: int,
        g: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Create a layout with one dominant splash panel (60%+ of page).

        The splash panel gets the majority of the page, with remaining panels
        arranged in the leftover space.

        Args:
            num_panels: Total number of panels
            splash_idx: Index of the splash panel (0-based)
            W: Available width (after margins)
            H: Available height (after margins)
            g: Gutter size

        Returns:
            List of (x, y, width, height) tuples
        """
        layout = [None] * num_panels
        other_panels = [i for i in range(num_panels) if i != splash_idx]

        if num_panels == 2:
            # 2 panels: splash on top (65%), other on bottom (35%)
            splash_h = int(H * 0.65)
            other_h = H - splash_h - g

            if splash_idx == 0:
                layout[0] = (0, 0, W, splash_h)
                layout[1] = (0, splash_h + g, W, other_h)
            else:
                layout[0] = (0, 0, W, other_h)
                layout[1] = (0, other_h + g, W, splash_h)

        elif num_panels == 3:
            # 3 panels: splash takes 60% height, other 2 split bottom row
            splash_h = int(H * 0.60)
            other_h = H - splash_h - g
            w_half = (W - g) // 2

            if splash_idx == 0:
                layout[0] = (0, 0, W, splash_h)
                layout[1] = (0, splash_h + g, w_half, other_h)
                layout[2] = (w_half + g, splash_h + g, w_half, other_h)
            elif splash_idx == 2:
                # Splash at bottom (cliffhanger style)
                layout[0] = (0, 0, w_half, other_h)
                layout[1] = (w_half + g, 0, w_half, other_h)
                layout[2] = (0, other_h + g, W, splash_h)
            else:
                # Middle panel is splash - unusual but handle it
                layout[0] = (0, 0, W, other_h // 2)
                layout[1] = (0, other_h // 2 + g, W, splash_h)
                layout[2] = (0, other_h // 2 + splash_h + 2 * g, W, other_h // 2)

        elif num_panels == 4:
            # 4 panels: splash takes 60% on left or top, others arranged in remaining space
            splash_w = int(W * 0.60)
            other_w = W - splash_w - g
            h_third = (H - 2 * g) // 3

            if splash_idx == 0:
                # Splash on left, 3 panels stacked on right
                layout[0] = (0, 0, splash_w, H)
                for i, idx in enumerate(other_panels):
                    layout[idx] = (splash_w + g, i * (h_third + g), other_w, h_third)
            elif splash_idx == 3:
                # Splash at bottom-right (cliffhanger)
                h_top = int(H * 0.40)
                h_bottom = H - h_top - g
                w_half = (W - g) // 2
                layout[0] = (0, 0, w_half, h_top)
                layout[1] = (w_half + g, 0, w_half, h_top)
                layout[2] = (0, h_top + g, w_half, h_bottom)
                layout[3] = (w_half + g, h_top + g, w_half, h_bottom)
                # Make splash larger by adjusting
                layout[3] = (int(W * 0.35) + g, h_top + g, int(W * 0.65) - g, h_bottom)
                layout[2] = (0, h_top + g, int(W * 0.35), h_bottom)
            else:
                # Default: splash on top spanning full width
                splash_h = int(H * 0.55)
                other_h = H - splash_h - g
                w_third = (W - 2 * g) // 3
                layout[splash_idx] = (0, 0, W, splash_h)
                remaining = [i for i in range(num_panels) if i != splash_idx]
                for i, idx in enumerate(remaining):
                    layout[idx] = (i * (w_third + g), splash_h + g, w_third, other_h)

        else:
            # 5+ panels: splash takes top 55%, others in grid below
            splash_h = int(H * 0.55)
            other_h = H - splash_h - g

            layout[splash_idx] = (0, 0, W, splash_h)

            # Arrange remaining panels in rows below
            remaining = other_panels
            cols = min(3, len(remaining))
            rows = (len(remaining) + cols - 1) // cols
            cell_w = (W - (cols - 1) * g) // cols
            cell_h = (other_h - (rows - 1) * g) // rows

            for i, idx in enumerate(remaining):
                row = i // cols
                col = i % cols
                x = col * (cell_w + g)
                y = splash_h + g + row * (cell_h + g)
                layout[idx] = (x, y, cell_w, cell_h)

        return layout

    def _action_grid_layout(
        self,
        num_panels: int,
        W: int,
        H: int,
        g: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Tighter grid layout for rapid action sequences (many small panels).

        Creates a denser grid with more panels per row for fast-paced scenes.

        Args:
            num_panels: Number of panels
            W: Available width
            H: Available height
            g: Gutter size

        Returns:
            List of (x, y, width, height) tuples
        """
        layout = []

        if num_panels <= 4:
            # 2x2 grid
            cols, rows = 2, 2
        elif num_panels <= 6:
            # 3x2 grid
            cols, rows = 3, 2
        elif num_panels <= 9:
            # 3x3 grid
            cols, rows = 3, 3
        else:
            # 4x3 or more
            cols = 4
            rows = (num_panels + cols - 1) // cols

        cell_w = (W - (cols - 1) * g) // cols
        cell_h = (H - (rows - 1) * g) // rows

        for i in range(num_panels):
            row = i // cols
            col = i % cols
            x = col * (cell_w + g)
            y = row * (cell_h + g)
            layout.append((x, y, cell_w, cell_h))

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
