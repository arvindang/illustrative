import os
import json
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

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

    def draw_speech_bubble(self, draw, text, position):
        """Draws a rounded speech bubble with wrapped text."""
        x, y = position
        wrapped_text = self.wrap_text(text, 500)
        
        # Calculate text size for bubble dimensions
        bbox = draw.multiline_textbbox((x, y), wrapped_text, font=self.font)
        padding = 20
        bubble_rect = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
        
        # Draw bubble background (White with black outline)
        draw.rounded_rectangle(bubble_rect, radius=15, fill="white", outline="black", width=3)
        
        # Draw the tail of the bubble (a small triangle)
        tail_coords = [(bubble_rect[0]+20, bubble_rect[3]), (bubble_rect[0]+40, bubble_rect[3]+30), (bubble_rect[0]+60, bubble_rect[3])]
        draw.polygon(tail_coords, fill="white", outline="black")
        # Overwrite the outline on the top of the tail to merge with the bubble
        draw.line([(tail_coords[0][0]+2, tail_coords[0][1]), (tail_coords[2][0]-2, tail_coords[2][1])], fill="white", width=4)

        # Draw the text
        draw.multiline_text((x, y), wrapped_text, font=self.font, fill="black")

    def assemble_page(self, page_data):
        page_num = page_data['page_number']
        print(f"📄 Assembling Page {page_num}...")
        
        # Create a blank white canvas
        canvas = Image.new("RGB", (self.page_width, self.page_height), "white")
        draw = ImageDraw.Draw(canvas)
        
        # Calculate panel sizes for a 2x2 grid
        panel_w = (self.page_width - (2 * self.margin) - self.gutter) // 2
        panel_h = (self.page_height - (2 * self.margin) - self.gutter) // 2
        
        for idx, panel in enumerate(page_data['panels']):
            panel_id = panel['panel_id']
            img_path = self.panels_dir / f"page_{page_num}" / f"panel_{panel_id}.png"
            
            if not img_path.exists():
                print(f"⚠️ Missing image for Page {page_num}, Panel {panel_id}")
                continue
                
            # Load and resize the generated panel
            # Use ImageOps.fit to maintain aspect ratio (center crop)
            panel_img = Image.open(img_path)
            panel_img = ImageOps.fit(panel_img, (panel_w, panel_h), method=Image.Resampling.LANCZOS)
            
            # Calculate position on the grid
            col = idx % 2
            row = idx // 2
            pos_x = self.margin + (col * (panel_w + self.gutter))
            pos_y = self.margin + (row * (panel_h + self.gutter))
            
            # Paste the panel
            canvas.paste(panel_img, (pos_x, pos_y))
            
            # Draw the dialogue bubble (Positioned at the top-left of the panel)
            if panel['dialogue']:
                bubble_pos = (pos_x + 40, pos_y + 40)
                self.draw_speech_bubble(draw, panel['dialogue'], bubble_pos)

        # Save the final page
        output_path = self.output_dir / f"page_{page_num}.png"
        canvas.save(output_path)
        print(f"✅ Page {page_num} saved to {output_path}")

    def run(self):
        with open(self.script_path, "r") as f:
            script_data = json.load(f)
            
        for page in script_data:
            self.assemble_page(page)

if __name__ == "__main__":
    compositor = CompositorAgent("assets/output/20-thousand-leagues-under-the-sea_test_page.json")
    compositor.run()