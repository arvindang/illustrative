import os
import json
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from exporter_agent import ExporterAgent

class CompositorAgent:
    def __init__(self, script_path: str):
        self.script_path = Path(script_path)
        self.panels_dir = Path("assets/output/pages")
        self.output_dir = Path("assets/output/final_pages")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration for a standard 2x2 grid
        self.page_width = 2400
        self.page_height = 3200
        self.margin = 60
        self.gutter = 40
        
        # Load a font (Ensure you have a .ttf file in a 'fonts' folder)
        # Fallback to a basic system font if not found
        try:
            self.font = ImageFont.truetype("fonts/PatrickHand-Regular.ttf", 60)
        except:
            print("⚠️ Font not found, using default.")
            self.font = ImageFont.load_default()

    def wrap_text(self, text, max_width):
        """Wraps text to fit within a specified width."""
        # Estimate characters per line based on font size (rough approximation)
        avg_char_width = 25 
        chars_per_line = max(1, int(max_width / avg_char_width))
        return textwrap.fill(text, width=chars_per_line)

    def draw_speech_bubble(self, draw, text, panel_rect, position_code="top-left"):
        """Draws a rounded speech bubble with wrapped text positioned dynamically."""
        panel_x, panel_y, panel_w, panel_h = panel_rect
        wrapped_text = self.wrap_text(text, 500)
        
        # Calculate text size
        # We use (0,0) to get dimensions relative to the anchor point.
        # We rely on bbox[2] (right) and bbox[3] (bottom) to ensure we cover the full rendered area.
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=self.font)
        text_w = bbox[2]
        text_h = bbox[3]
        
        padding = 30
        bubble_w = text_w + (padding * 2)
        bubble_h = text_h + (padding * 2)
        
        # Determine coordinates based on position_code
        # Default margin from panel edges
        edge_margin = 40 
        
        if position_code == "top-right":
            x = panel_x + panel_w - bubble_w - edge_margin
            y = panel_y + edge_margin
        elif position_code == "bottom-left":
            x = panel_x + edge_margin
            y = panel_y + panel_h - bubble_h - edge_margin
        elif position_code == "bottom-right":
            x = panel_x + panel_w - bubble_w - edge_margin
            y = panel_y + panel_h - bubble_h - edge_margin
        else: # top-left (default)
            x = panel_x + edge_margin
            y = panel_y + edge_margin

        # Draw bubble background
        bubble_rect = [x, y, x + bubble_w, y + bubble_h]
        draw.rounded_rectangle(bubble_rect, radius=15, fill="white", outline="black", width=3)
        
        # Draw text centered in bubble
        text_x = x + padding
        text_y = y + padding
        draw.multiline_text((text_x, text_y), wrapped_text, font=self.font, fill="black")

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
        
        # Get dynamic layout coordinates
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
                
                # Draw the dialogue bubble
                if panel.get('dialogue'):
                    # Get preferred position from JSON, default to top-left
                    pos_code = panel.get('bubble_position', 'top-left')
                    panel_rect = (pos_x, pos_y, panel_w, panel_h)
                    self.draw_speech_bubble(draw, panel['dialogue'], panel_rect, pos_code)
                    
            except Exception as e:
                print(f"❌ Error processing Panel {panel_id}: {e}")

        # Save the final page
        output_path = self.output_dir / f"page_{page_num}.png"
        canvas.save(output_path)
        print(f"✅ Page {page_num} saved to {output_path}")

    def run(self):
        with open(self.script_path, "r") as f:
            script_data = json.load(f)
            
        for page in script_data:
            self.assemble_page(page)
        
        # Packaging
        print("📦 Packaging output...")
        output_base = Path("assets/output") / self.script_path.stem
        exporter = ExporterAgent(str(self.output_dir), str(output_base))
        exporter.export_pdf()
        exporter.export_epub(title=self.script_path.stem.replace("-", " ").title())

if __name__ == "__main__":
    script_path = "assets/output/20-thousand-leagues-under-the-sea_test_page.json"
    print(f"📖 Loading script: {script_path}")

    compositor = CompositorAgent(script_path)
    compositor.run()