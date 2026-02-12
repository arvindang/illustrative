"""
10-page OpenAI batch stress test.

Generates ~38 panels across 10 pages covering key scenes from
20,000 Leagues Under the Sea. Tests batch scaling, multi-page composition,
and PDF/EPUB export at volume.

Usage:
    python tests/test_openai_10page.py
"""
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

STYLE = "Lush Watercolor"
TONE = "Philosophical"

CHARACTERS = [
    {
        "name": "Captain Nemo",
        "aliases": ["Nemo", "The Captain"],
        "description": (
            "A tall, enigmatic man in his 40s with dark piercing eyes, olive skin, "
            "and a neatly trimmed black beard streaked with silver. Wears an ornate "
            "navy-blue naval uniform with gold epaulettes, brass buttons, and a high "
            "collar. Carries himself with regal authority. Expression shifts between "
            "cold detachment and passionate intensity."
        ),
        "age_range": "40s",
        "occupation": "Captain of the Nautilus",
        "distinctive_items": ["ornate naval uniform", "brass telescope", "gold ring"],
        "color_signature": "#1B3A5C",
    },
    {
        "name": "Professor Aronnax",
        "aliases": ["Aronnax", "Pierre Aronnax", "The Professor"],
        "description": (
            "A scholarly French naturalist in his 50s with grey-streaked brown hair, "
            "round wire-rimmed spectacles, and a kind but perpetually curious expression. "
            "Wears a brown tweed jacket over a white shirt with a silk cravat. Lean build, "
            "ink-stained fingers, always carrying his leather-bound journal."
        ),
        "age_range": "50s",
        "occupation": "Marine Biologist / Professor at the Paris Museum",
        "distinctive_items": ["wire-rimmed spectacles", "leather journal", "ink pen"],
        "color_signature": "#8B6914",
    },
    {
        "name": "Ned Land",
        "aliases": ["Ned", "The Canadian", "Master Land"],
        "description": (
            "A powerfully built Canadian harpooner in his late 30s with a thick reddish-brown "
            "beard, weathered tan skin, and fierce blue eyes. Broad-shouldered and tall. "
            "Wears a worn leather vest over a white linen shirt, canvas trousers, and heavy "
            "boots. Often has a harpoon slung across his back."
        ),
        "age_range": "late 30s",
        "occupation": "Master Harpooner",
        "distinctive_items": ["harpoon", "leather vest", "thick beard"],
        "color_signature": "#8B4513",
    },
    {
        "name": "Conseil",
        "aliases": ["Conseil"],
        "description": (
            "A neat, composed Flemish man in his early 30s with sandy blond hair parted "
            "precisely, a clean-shaven face, and calm grey eyes. Wears a well-pressed dark "
            "waistcoat and white shirt even in the most trying circumstances. Thin build, "
            "always composed and formal in bearing."
        ),
        "age_range": "early 30s",
        "occupation": "Aronnax's loyal manservant and taxonomist",
        "distinctive_items": ["pressed waistcoat", "classification notebook"],
        "color_signature": "#556B2F",
    },
]

OBJECTS = [
    {
        "name": "The Nautilus",
        "description": (
            "A magnificent Victorian-era submarine shaped like an elongated spindle, "
            "70 meters long. Riveted iron hull with large circular portholes emitting "
            "a warm amber glow. The interior is lavishly decorated with mahogany panels, "
            "velvet drapes, brass instruments, and a pipe organ in the salon."
        ),
        "key_features": [
            "riveted iron hull",
            "large circular portholes with amber glow",
            "ornate Victorian interior with pipe organ",
            "sleek spindle shape",
        ],
        "condition": "pristine, well-maintained",
        "material_context": "1860s industrial iron, brass, and mahogany",
    },
    {
        "name": "Diving Suits",
        "description": (
            "Heavy Victorian-era diving apparatus: thick canvas suits reinforced with "
            "riveted copper plates, large spherical copper helmets with glass visors, "
            "air hoses connected to back-mounted tanks, weighted boots."
        ),
        "key_features": [
            "copper helmet with glass visor",
            "canvas and copper body suit",
            "air tank backpack",
            "weighted magnetic boots",
        ],
        "condition": "well-maintained, polished copper",
        "material_context": "1860s diving technology, copper and canvas",
    },
]

SCRIPT = [
    # PAGE 1: The Hunt Begins
    {
        "page_number": 1,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "A dramatic wide shot of the USS Abraham Lincoln, a large American "
                    "warship, cutting through stormy grey seas. Dark clouds loom overhead. "
                    "Sailors crowd the rails, scanning the horizon with telescopes."
                ),
                "characters": [],
                "dialogue": "",
                "caption": "1867. The world's navies hunted a creature that had terrorized the seas.",
                "bubble_position": "top-left",
                "shot_type": "establishing",
                "panel_size": "large",
                "key_objects": [],
                "advice": {"scene_type": "establishing", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "Medium shot of Professor Aronnax standing at the ship's bow, wind "
                    "whipping his jacket, gripping the railing. His journal is tucked under "
                    "one arm. He stares out at the ocean with scientific fascination."
                ),
                "characters": ["Professor Aronnax"],
                "dialogue": "",
                "caption": "Professor Pierre Aronnax, naturalist, had been invited aboard to identify the beast.",
                "bubble_position": "bottom-right",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "introduction", "composition": {"negative_space": "bottom-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "Close-up of Ned Land sharpening his harpoon on the deck, muscles "
                    "taut, a confident grin on his weathered face. Conseil stands behind "
                    "him, hands clasped, looking mildly concerned."
                ),
                "characters": ["Ned Land", "Conseil"],
                "dialogue": "When I find this monster, one throw is all I'll need.",
                "caption": "",
                "bubble_position": "top-right",
                "shot_type": "close-up",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "dialogue", "composition": {"negative_space": "top-right"}},
            },
            {
                "panel_id": 4,
                "visual_description": (
                    "Night scene. A massive phosphorescent glow erupts from beneath the "
                    "dark ocean surface. The sailors on deck point and shout in terror. "
                    "The glow illuminates the underside of the ship in eerie blue-green."
                ),
                "characters": [],
                "dialogue": "",
                "caption": "Then, on the third night, the sea itself came alive.",
                "bubble_position": "bottom-left",
                "shot_type": "wide",
                "panel_size": "large",
                "key_objects": [],
                "advice": {"scene_type": "action", "composition": {"negative_space": "bottom-left"}},
            },
        ],
    },
    # PAGE 2: Cast Into the Sea
    {
        "page_number": 2,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "Chaotic action panel: a massive collision rocks the warship. "
                    "Water crashes over the deck. Professor Aronnax, Ned Land, and "
                    "Conseil are thrown overboard into dark churning waves. The ship "
                    "lists dangerously to one side."
                ),
                "characters": ["Professor Aronnax", "Ned Land", "Conseil"],
                "dialogue": "",
                "caption": "The impact threw us into the black Atlantic.",
                "bubble_position": "top-left",
                "shot_type": "wide",
                "panel_size": "large",
                "key_objects": [],
                "advice": {"scene_type": "action", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "Underwater view looking up: three silhouettes struggle at the surface, "
                    "moonlight filtering through the waves. Below them, the dark shape of "
                    "something enormous glides past."
                ),
                "characters": ["Professor Aronnax", "Ned Land", "Conseil"],
                "dialogue": "",
                "caption": "We clung to each other in the darkness, and beneath us, something moved.",
                "bubble_position": "bottom-right",
                "shot_type": "worms-eye",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "suspense", "composition": {"negative_space": "bottom-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "Close-up of Ned Land's hand slamming against a smooth metal surface "
                    "rising from the water. His eyes are wide with shock. Rivets and iron "
                    "plates glisten in the moonlight. Not a whale — a machine."
                ),
                "characters": ["Ned Land"],
                "dialogue": "This is no animal. It's iron!",
                "caption": "",
                "bubble_position": "top-right",
                "shot_type": "extreme-close-up",
                "panel_size": "medium",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "revelation", "composition": {"negative_space": "top-right"}},
            },
            {
                "panel_id": 4,
                "visual_description": (
                    "A hatch opens on the surface of the iron vessel, warm amber light "
                    "spilling out. Dark figures emerge and pull the three men inside. "
                    "The Nautilus's hull stretches into the distance."
                ),
                "characters": ["Professor Aronnax", "Ned Land", "Conseil"],
                "dialogue": "",
                "caption": "We were taken aboard the impossible vessel.",
                "bubble_position": "bottom-left",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "transition", "composition": {"negative_space": "bottom-left"}},
            },
        ],
    },
    # PAGE 3: Meeting Captain Nemo
    {
        "page_number": 3,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "The luxurious salon of the Nautilus. Floor-to-ceiling bookshelves, "
                    "oil paintings in gilt frames, a pipe organ against one wall, glass "
                    "display cases of marine specimens. Warm brass lamplight. The three "
                    "captives stand together, dripping wet, looking around in astonishment."
                ),
                "characters": ["Professor Aronnax", "Ned Land", "Conseil"],
                "dialogue": "",
                "caption": "Nothing could have prepared us for what lay within.",
                "bubble_position": "top-left",
                "shot_type": "establishing",
                "panel_size": "large",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "establishing", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "Captain Nemo enters through an arched doorway, silhouetted against "
                    "the light. He is imposing, hands clasped behind his back, chin raised. "
                    "His gold epaulettes catch the lamplight."
                ),
                "characters": ["Captain Nemo"],
                "dialogue": "",
                "caption": "Then he appeared — the master of this vessel.",
                "bubble_position": "bottom-right",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "introduction", "composition": {"negative_space": "bottom-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "Two-shot of Nemo and Aronnax face-to-face. Nemo's expression is cold "
                    "and appraising. Aronnax meets his gaze with cautious respect. "
                    "Behind them, the salon's enormous porthole shows deep blue ocean."
                ),
                "characters": ["Captain Nemo", "Professor Aronnax"],
                "dialogue": "I am Captain Nemo. And you are aboard the Nautilus.",
                "caption": "",
                "bubble_position": "top-right",
                "shot_type": "two-shot",
                "panel_size": "medium",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "dialogue", "composition": {"negative_space": "top-right"}},
            },
            {
                "panel_id": 4,
                "visual_description": (
                    "Close-up of Ned Land glaring at Nemo with barely contained fury, "
                    "his fist clenched. Conseil places a restraining hand on Ned's arm."
                ),
                "characters": ["Ned Land", "Conseil"],
                "dialogue": "Prisoners, then. That's what we are.",
                "caption": "",
                "bubble_position": "bottom-left",
                "shot_type": "close-up",
                "panel_size": "small",
                "key_objects": [],
                "advice": {"scene_type": "tension", "composition": {"negative_space": "bottom-left"}},
            },
        ],
    },
    # PAGE 4: The Underwater Window
    {
        "page_number": 4,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "Splash page: the enormous observation window of the Nautilus reveals "
                    "a breathtaking underwater panorama. A vast coral reef in vivid reds, "
                    "purples, and golds. Schools of tropical fish swirl in spirals. "
                    "Aronnax presses his hands against the glass, mouth open in wonder. "
                    "Nemo stands behind him, arms crossed, watching the professor's reaction."
                ),
                "characters": ["Professor Aronnax", "Captain Nemo"],
                "dialogue": "The sea is everything. It covers seven-tenths of the globe.",
                "caption": "",
                "bubble_position": "top-left",
                "shot_type": "establishing",
                "panel_size": "large",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "wonder", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "Close-up of Aronnax's face reflected in the glass, his spectacles "
                    "catching the bioluminescent glow of jellyfish drifting past. Tears "
                    "of wonder in his eyes."
                ),
                "characters": ["Professor Aronnax"],
                "dialogue": "",
                "caption": "In that moment, I forgot I was a prisoner.",
                "bubble_position": "bottom-right",
                "shot_type": "extreme-close-up",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "introspection", "composition": {"negative_space": "bottom-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "Ned Land sits alone in a spartan cabin, staring at the iron wall. "
                    "His harpoon leans against the corner. His expression is brooding, "
                    "fists on his knees. A small porthole shows only darkness."
                ),
                "characters": ["Ned Land"],
                "dialogue": "",
                "caption": "But Ned Land saw only a cage.",
                "bubble_position": "bottom-left",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "contrast", "composition": {"negative_space": "bottom-left"}},
            },
        ],
    },
    # PAGE 5: The Underwater Walk
    {
        "page_number": 5,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "Nemo, Aronnax, and Conseil in heavy Victorian diving suits with "
                    "copper helmets, stepping out of the Nautilus's airlock onto the "
                    "ocean floor. Shafts of green light filter down from above. "
                    "The sandy seabed stretches ahead dotted with sea anemones."
                ),
                "characters": ["Captain Nemo", "Professor Aronnax", "Conseil"],
                "dialogue": "",
                "caption": "Captain Nemo invited us to walk upon the ocean floor.",
                "bubble_position": "top-left",
                "shot_type": "wide",
                "panel_size": "large",
                "key_objects": ["Diving Suits"],
                "advice": {"scene_type": "adventure", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "Aronnax kneels beside a massive coral formation, examining it "
                    "through his helmet visor. He gestures excitedly to Conseil, who "
                    "takes careful notes on a waterproof slate. Tropical fish surround them."
                ),
                "characters": ["Professor Aronnax", "Conseil"],
                "dialogue": "",
                "caption": "Every step revealed specimens unknown to science.",
                "bubble_position": "top-right",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": ["Diving Suits"],
                "advice": {"scene_type": "discovery", "composition": {"negative_space": "top-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "Nemo raises his hand in warning. Ahead of them, a giant shark — "
                    "at least 8 meters long — cruises slowly through the murky water, "
                    "its eye catching the light. The three divers freeze."
                ),
                "characters": ["Captain Nemo", "Professor Aronnax", "Conseil"],
                "dialogue": "",
                "caption": "Then the sea reminded us we were guests in a predator's realm.",
                "bubble_position": "bottom-left",
                "shot_type": "wide",
                "panel_size": "large",
                "key_objects": ["Diving Suits"],
                "advice": {"scene_type": "danger", "composition": {"negative_space": "bottom-left"}},
            },
        ],
    },
    # PAGE 6: The Pearl Diver
    {
        "page_number": 6,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "An underwater grotto bathed in soft blue-green light. Captain Nemo "
                    "leads the group toward an enormous oyster, its shell partially open "
                    "to reveal a pearl the size of a coconut, glowing with iridescence."
                ),
                "characters": ["Captain Nemo", "Professor Aronnax"],
                "dialogue": "",
                "caption": "Nemo showed us treasures the surface world could only dream of.",
                "bubble_position": "top-left",
                "shot_type": "medium",
                "panel_size": "large",
                "key_objects": ["Diving Suits"],
                "advice": {"scene_type": "wonder", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "A young Indian pearl diver is attacked by a shark. He thrashes in "
                    "the water, blood trailing. The scene is terrifying and chaotic."
                ),
                "characters": [],
                "dialogue": "",
                "caption": "A scream — though underwater, we felt it.",
                "bubble_position": "top-right",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "action", "composition": {"negative_space": "top-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "Captain Nemo charges at the shark with a knife, plunging it into "
                    "the creature's belly. His diving suit is scratched, his expression "
                    "behind the copper visor is fierce. Bubbles explode around the struggle."
                ),
                "characters": ["Captain Nemo"],
                "dialogue": "",
                "caption": "",
                "bubble_position": "bottom-left",
                "shot_type": "close-up",
                "panel_size": "large",
                "key_objects": ["Diving Suits"],
                "advice": {"scene_type": "action", "composition": {"negative_space": "bottom-left"}},
            },
            {
                "panel_id": 4,
                "visual_description": (
                    "Nemo cradles the unconscious pearl diver, carrying him toward the "
                    "surface. Aronnax watches in astonishment. Nemo's eyes behind the "
                    "visor show compassion — a side of him unseen before."
                ),
                "characters": ["Captain Nemo", "Professor Aronnax"],
                "dialogue": "",
                "caption": "This man who shunned humanity... saved a stranger's life.",
                "bubble_position": "bottom-right",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": ["Diving Suits"],
                "advice": {"scene_type": "emotional", "composition": {"negative_space": "bottom-right"}},
            },
        ],
    },
    # PAGE 7: Trapped Under the Ice
    {
        "page_number": 7,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "The Nautilus trapped beneath a massive ice shelf at the South Pole. "
                    "The submarine is wedged between towering walls of pale blue ice. "
                    "Cracks spider-web across the frozen ceiling above. The scene is "
                    "claustrophobic and cold."
                ),
                "characters": [],
                "dialogue": "",
                "caption": "At the bottom of the world, the ice closed around us like a fist.",
                "bubble_position": "top-left",
                "shot_type": "establishing",
                "panel_size": "large",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "danger", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "Inside the Nautilus, Nemo studies charts at the navigation table. "
                    "His face is drawn and tense. Aronnax and Conseil look worried. "
                    "Condensation drips from the brass fittings — the air is running out."
                ),
                "characters": ["Captain Nemo", "Professor Aronnax", "Conseil"],
                "dialogue": "We have forty-eight hours of air. Perhaps less.",
                "caption": "",
                "bubble_position": "top-right",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "tension", "composition": {"negative_space": "top-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "The entire crew of the Nautilus — muscular, diverse men in work "
                    "clothes — hack at the ice with picks and axes, working in shifts. "
                    "Steaming hot water is pumped from the Nautilus onto the ice walls. "
                    "Exhaustion is visible on every face."
                ),
                "characters": ["Captain Nemo"],
                "dialogue": "",
                "caption": "Every man worked until he collapsed. Then the next man took his place.",
                "bubble_position": "bottom-left",
                "shot_type": "wide",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "action", "composition": {"negative_space": "bottom-left"}},
            },
            {
                "panel_id": 4,
                "visual_description": (
                    "The Nautilus breaks through the ice into open water, chunks of ice "
                    "exploding outward. Brilliant sunlight streams down through the gap. "
                    "Inside, the crew collapses with relief."
                ),
                "characters": [],
                "dialogue": "",
                "caption": "Freedom. We gasped it in like men reborn.",
                "bubble_position": "bottom-right",
                "shot_type": "wide",
                "panel_size": "large",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "triumph", "composition": {"negative_space": "bottom-right"}},
            },
        ],
    },
    # PAGE 8: The Sunken City (Atlantis)
    {
        "page_number": 8,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "A breathtaking underwater panorama of ruins — crumbled columns, "
                    "fallen temples, mosaic floors covered in barnacles and swaying kelp. "
                    "The ruins stretch as far as the eye can see, lit by volcanic vents "
                    "that glow orange beneath the ancient stones. This is Atlantis."
                ),
                "characters": [],
                "dialogue": "",
                "caption": "Atlantis. The legend was real, and it slept beneath the waves.",
                "bubble_position": "top-left",
                "shot_type": "establishing",
                "panel_size": "large",
                "key_objects": [],
                "advice": {"scene_type": "wonder", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "Aronnax in his diving suit walks through a colonnade of broken "
                    "Greek-style columns, reaching out to touch ancient carvings. "
                    "Fish swim through empty windows of what was once a grand building. "
                    "His face behind the visor shows pure awe."
                ),
                "characters": ["Professor Aronnax"],
                "dialogue": "",
                "caption": "I walked where kings had walked ten thousand years before.",
                "bubble_position": "bottom-right",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": ["Diving Suits"],
                "advice": {"scene_type": "wonder", "composition": {"negative_space": "bottom-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "Nemo stands atop a volcanic ridge overlooking the ruins, arms "
                    "spread wide. Lava glows beneath his boots. His silhouette is "
                    "framed against the vast underwater city. He looks like a god "
                    "surveying his domain."
                ),
                "characters": ["Captain Nemo"],
                "dialogue": "",
                "caption": "Here, Captain Nemo was more than a man. He was a sovereign of the deep.",
                "bubble_position": "bottom-left",
                "shot_type": "wide",
                "panel_size": "large",
                "key_objects": ["Diving Suits"],
                "advice": {"scene_type": "dramatic", "composition": {"negative_space": "bottom-left"}},
            },
        ],
    },
    # PAGE 9: The Giant Squid
    {
        "page_number": 9,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "Horror scene: enormous tentacles — each thick as a tree trunk — "
                    "wrap around the hull of the Nautilus. Suction cups the size of "
                    "dinner plates grip the iron. Through the observation window, "
                    "a massive eye stares in — the giant squid."
                ),
                "characters": [],
                "dialogue": "",
                "caption": "The kraken. It was real.",
                "bubble_position": "top-left",
                "shot_type": "establishing",
                "panel_size": "large",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "horror", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "On the deck of the surfaced Nautilus, the crew fights the squid "
                    "with axes and harpoons. Ned Land drives his harpoon deep into a "
                    "writhing tentacle. Spray and ink fill the air. Chaos and courage."
                ),
                "characters": ["Ned Land"],
                "dialogue": "Come on then, beast!",
                "caption": "",
                "bubble_position": "top-right",
                "shot_type": "wide",
                "panel_size": "large",
                "key_objects": [],
                "advice": {"scene_type": "action", "composition": {"negative_space": "top-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "A crewman is lifted into the air by a tentacle, screaming. "
                    "Captain Nemo slashes at the tentacle with a cutlass, his face "
                    "twisted in desperate fury. Aronnax and Conseil pull at the tentacle."
                ),
                "characters": ["Captain Nemo", "Professor Aronnax", "Conseil"],
                "dialogue": "",
                "caption": "We fought together — prisoner and captor, united against the deep.",
                "bubble_position": "bottom-left",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "action", "composition": {"negative_space": "bottom-left"}},
            },
            {
                "panel_id": 4,
                "visual_description": (
                    "Aftermath. The squid retreats into dark water trailing ink. "
                    "The deck is scarred and slick. Nemo kneels beside a fallen crewman, "
                    "head bowed. A single tear runs down his cheek."
                ),
                "characters": ["Captain Nemo", "Ned Land"],
                "dialogue": "",
                "caption": "We won. But not all of us survived.",
                "bubble_position": "bottom-right",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": [],
                "advice": {"scene_type": "emotional", "composition": {"negative_space": "bottom-right"}},
            },
        ],
    },
    # PAGE 10: The Escape / Maelstrom
    {
        "page_number": 10,
        "panels": [
            {
                "panel_id": 1,
                "visual_description": (
                    "Night. Ned Land, Aronnax, and Conseil creep through the dark "
                    "corridors of the Nautilus toward the small dinghy. Emergency "
                    "red lighting pulses. They move with desperate urgency."
                ),
                "characters": ["Ned Land", "Professor Aronnax", "Conseil"],
                "dialogue": "Now. It has to be now.",
                "caption": "",
                "bubble_position": "top-left",
                "shot_type": "medium",
                "panel_size": "medium",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "suspense", "composition": {"negative_space": "top-left"}},
            },
            {
                "panel_id": 2,
                "visual_description": (
                    "Aronnax pauses at Nemo's door, hand raised as if to knock. "
                    "Through the crack, he sees Nemo playing the pipe organ alone "
                    "in the salon, a figure of profound solitude. Music seems to "
                    "emanate from the scene."
                ),
                "characters": ["Professor Aronnax", "Captain Nemo"],
                "dialogue": "",
                "caption": "I almost stayed. God help me, I almost stayed.",
                "bubble_position": "bottom-right",
                "shot_type": "over-shoulder",
                "panel_size": "medium",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "emotional", "composition": {"negative_space": "bottom-right"}},
            },
            {
                "panel_id": 3,
                "visual_description": (
                    "The Maelstrom — a colossal whirlpool churning under a stormy sky. "
                    "The Nautilus is caught in its pull. The tiny dinghy with our three "
                    "heroes is tossed like a leaf. Waves tower above them. Lightning "
                    "cracks the sky."
                ),
                "characters": ["Ned Land", "Professor Aronnax", "Conseil"],
                "dialogue": "",
                "caption": "",
                "bubble_position": "top-right",
                "shot_type": "birds-eye",
                "panel_size": "large",
                "key_objects": ["The Nautilus"],
                "advice": {"scene_type": "climax", "composition": {"negative_space": "top-right"}},
            },
            {
                "panel_id": 4,
                "visual_description": (
                    "Calm. Dawn light over a peaceful Norwegian beach. Aronnax lies "
                    "on the sand, alive, his journal clutched to his chest. Ned and "
                    "Conseil lie nearby. The sea is serene. No sign of the Nautilus."
                ),
                "characters": ["Professor Aronnax", "Ned Land", "Conseil"],
                "dialogue": "",
                "caption": (
                    "Captain Nemo — wherever you are — the sea keeps your secrets. "
                    "But I will tell the world what I have seen."
                ),
                "bubble_position": "bottom-left",
                "shot_type": "wide",
                "panel_size": "large",
                "key_objects": [],
                "advice": {"scene_type": "resolution", "composition": {"negative_space": "bottom-left"}},
            },
        ],
    },
]


ASSETS = {
    "characters": CHARACTERS,
    "objects": OBJECTS,
    "color_script": {
        "act_1": {"palette": ["storm grey", "ocean blue", "moonlight silver"]},
        "act_2": {"palette": ["deep ocean blue", "amber lamplight", "coral red", "bioluminescent green"]},
        "act_3": {"palette": ["volcanic orange", "ink black", "dawn gold", "ice blue"]},
    },
}


async def main():
    total_panels = sum(len(p["panels"]) for p in SCRIPT)
    print("=" * 60)
    print("OPENAI 10-PAGE BATCH STRESS TEST")
    print("=" * 60)
    print(f"  Pages:         {len(SCRIPT)}")
    print(f"  Total panels:  {total_panels}")
    print(f"  Characters:    {len(CHARACTERS)}")
    print(f"  Objects:        {len(OBJECTS)}")
    print(f"  Backend:       {config.image_backend}")
    print(f"  Batch enabled: {config.openai_batch_enabled}")
    print(f"  Model:         {config.openai_image_model_primary}")
    print(f"  Panel quality: {config.openai_image_quality}")
    print(f"  Ref quality:   {config.openai_ref_image_quality}")
    print("=" * 60)

    if not config.openai_api_key:
        print("\nERROR: OPENAI_API_KEY not set.")
        return

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    base_dir = Path(f"assets/output/openai_10page_{timestamp}_{short_id}")
    base_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["characters", "objects", "pages", "final_pages"]:
        (base_dir / sub).mkdir(exist_ok=True)

    stem = "20K_Leagues_Under_the_Sea"
    script_path = base_dir / f"{stem}_full_script.json"
    assets_path = base_dir / f"{stem}_assets.json"

    with open(script_path, "w") as f:
        json.dump(SCRIPT, f, indent=2)
    with open(assets_path, "w") as f:
        json.dump(ASSETS, f, indent=2)

    print(f"\n  Output: {base_dir}\n")

    # --- STEP 1: References ---
    print("--- STEP 1: Reference sheets ---")
    from agents import get_image_agents
    style_prompt = f"{STYLE} style, {TONE} tone"
    ref_agent, panel_agent = get_image_agents(str(script_path), style_prompt, base_output_dir=base_dir)

    import time
    t0 = time.time()
    await ref_agent.generate_all_references(style=STYLE)
    ref_time = time.time() - t0

    char_count = len([d for d in (base_dir / "characters").iterdir() if d.is_dir()])
    obj_count = len([d for d in (base_dir / "objects").iterdir() if d.is_dir()])
    print(f"  Characters: {char_count}, Objects: {obj_count} ({ref_time:.0f}s)\n")

    # --- STEP 2: Panels ---
    print("--- STEP 2: Panel images ---")
    t1 = time.time()
    await panel_agent.run_production()
    panel_time = time.time() - t1

    panel_count = len(list((base_dir / "pages").rglob("*.png")))
    print(f"  Panels generated: {panel_count}/{total_panels} ({panel_time:.0f}s)\n")

    # --- STEP 3: Composition ---
    print("--- STEP 3: Composition + export ---")
    from agents import CompositorAgent
    t2 = time.time()
    compositor = CompositorAgent(str(script_path), base_output_dir=base_dir)
    compositor.run()
    comp_time = time.time() - t2

    final_count = len(list((base_dir / "final_pages").glob("*.png")))
    pdf_exists = (base_dir / f"{stem}_full_script.pdf").exists()
    epub_exists = (base_dir / f"{stem}_full_script.epub").exists()

    # --- RESULTS ---
    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("10-PAGE STRESS TEST RESULTS")
    print("=" * 60)
    print(f"  Characters:     {char_count}")
    print(f"  Objects:        {obj_count}")
    print(f"  Panels:         {panel_count}/{total_panels}")
    print(f"  Final pages:    {final_count}/{len(SCRIPT)}")
    print(f"  PDF exported:   {pdf_exists}")
    print(f"  EPUB exported:  {epub_exists}")
    print(f"")
    print(f"  Timing:")
    print(f"    References:   {ref_time:.0f}s")
    print(f"    Panels:       {panel_time:.0f}s")
    print(f"    Composition:  {comp_time:.0f}s")
    print(f"    Total:        {total_time:.0f}s")
    print(f"")
    print(f"  Output: {base_dir}")

    if panel_count == total_panels and final_count == len(SCRIPT):
        print(f"\n  PASSED - All {total_panels} panels and {len(SCRIPT)} pages generated!")
    elif panel_count > 0:
        print(f"\n  PARTIAL - {panel_count}/{total_panels} panels, {final_count}/{len(SCRIPT)} pages")
    else:
        print(f"\n  FAILED - No panels generated")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
