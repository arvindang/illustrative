import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ScriptingAgent:
    def __init__(self, book_path: str):
        self.book_path = Path(book_path)
        self.output_dir = Path("assets/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_content(self, test_mode=True):
        with open(self.book_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            if test_mode:
                # Take 5000 characters from the middle of the book for a "Random Page" test
                start_index = len(full_text) // 2
                
                # Attempt to find the start of a new line/paragraph to avoid cutting mid-sentence
                try:
                    # Find the next newline character
                    start_index = full_text.index('\\n', start_index) + 1
                except ValueError:
                    pass  # If no newline found, stick to the approximate middle

                return full_text[start_index : start_index + 5000]
            return full_text

    async def generate_script(self, style: str, tone: str, writing_style: str, test_mode=True):
        source_text = self.load_content(test_mode=test_mode)
        mode_label = "TEST PAGE" if test_mode else "FULL BOOK"
        
        print(f"🧠 Mode: {mode_label} | Analyzing: {self.book_path.name}...")

        prompt = f"""
        Act as a Graphic Novel Director specializing in the '{style}' aesthetic.
        The tone of this adaptation is '{tone}'.
        The writing style should be '{writing_style}'.

        When writing 'visual_description':
        - If style is 'Watercolor', focus on color bleeds and soft edges.
        - If style is 'Noir', focus on stark silhouettes and heavy ink.
        - If style is 'Botanical', focus on thin lines, flat colors, emphasize natural surroundings.         
        - Use camera terminology (Low Angle, Dutch Tilt, Close-up) that matches the '{tone}'.
        
        Adapt the following book into a structured graphic novel script.
        
        For each page, determine the optimal layout and number of panels (1-6) based on narrative pacing:
        - 1-2 Panels: Establishing shots, emotional beats, beautiful scenery, or dramatic reveals.
        - 3-4 Panels: Balanced storytelling, standard conversations.
        - 5-6 Panels: Quick action sequences, fast-paced dialogue, or montages.

        Ensure a logical reading flow (Z-pattern: Top-Left -> Top-Right -> Bottom-Left -> Bottom-Right).
        
        Each panel must include:
        1. visual_description: A detailed prompt for an AI image generator.
        2. dialogue: The text to appear in speech bubbles.
        3. character_presence: A list of main characters in the panel.
        4. bubble_position: Best location for text ("top-left", "top-right", "bottom-left", "bottom-right") to avoid covering faces/action.

        OUTPUT FORMAT: Provide only valid JSON.
        """

        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt, source_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "page_number": {"type": "INTEGER"},
                            "layout_style": {
                                "type": "STRING", 
                                "enum": ["cinematic_widescreen", "action_dynamic", "standard_grid", "dialogue_focus", "full_splash"],
                                "description": "The overall layout template to use for this page."
                            },
                            "panels": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "panel_id": {"type": "INTEGER"},
                                        "visual_description": {"type": "STRING"},
                                        "dialogue": {"type": "STRING"},
                                        "characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                                        "bubble_position": {"type": "STRING", "enum": ["top-left", "top-right", "bottom-left", "bottom-right"]}
                                    },
                                    "required": ["panel_id", "visual_description", "dialogue", "characters", "bubble_position"]
                                }
                            }
                        },
                        "required": ["page_number", "layout_style", "panels"]
                    }
                }
            )
        )

        # Save the result
        suffix = "_test_page.json" if test_mode else "_full_script.json"
        output_file = self.output_dir / f"{self.book_path.stem}{suffix}"
        
        with open(output_file, "w") as f:
            json.dump(response.parsed, f, indent=2)
        
        print(f"✅ Success! Preview your script here: {output_file}")
        return response.parsed

if __name__ == "__main__":
    import asyncio
    agent = ScriptingAgent("assets/input/20-thousand-leagues-under-the-sea.txt")
    # Change test_mode=False once you like the results of Page 1
    asyncio.run(agent.generate_script(style="Watercolor", tone="Melancholic, Flowing, Vast", writing_style="Cinematic, Jules Verne-esque but modern pacing", test_mode=True))

