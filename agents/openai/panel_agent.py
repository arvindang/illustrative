"""
OpenAI Panel Agent: Comic panel image generation.

Mirrors the PanelAgent interface from agents/panel_agent.py but uses
OpenAI's image API (batch or sync) instead of Gemini.

Supports:
- Batch mode: All panels submitted via batch API (50% savings, 24h SLA)
- Sync mode: Sequential generation with rate limiting (test/UI, instant)
- Character consistency via reference images (file_id in edits endpoint)
- Resume via ProductionManifest
- Model fallback chain
"""
import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from config import config
from utils import ProductionManifest, RateLimiter

from agents.openai.client import get_openai_client
from agents.openai.batch_manager import BatchRequest, OpenAIBatchManager
from agents.openai.file_manager import OpenAIFileManager
from agents.openai.reference_agent import OpenAIReferenceAgent


class OpenAIPanelAgent:
    """
    Generates comic panel images using OpenAI's image API.

    Drop-in replacement for PanelAgent when IMAGE_BACKEND=openai.
    Outputs to the same directory structure (pages/page_N/panel_N.png).
    """

    def __init__(
        self,
        script_path: str,
        style_prompt: str,
        base_output_dir: Path = None,
        reference_agent: OpenAIReferenceAgent = None,
    ):
        self.script_path = Path(script_path)
        self.style_prompt = style_prompt

        if base_output_dir:
            self.base_dir = Path(base_output_dir)
            self.output_base_dir = self.base_dir / "pages"
        else:
            self.base_dir = Path("assets/output")
            self.output_base_dir = Path("assets/output/pages")

        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Manifest for resume capability
        manifest_path = self.output_base_dir.parent / "production_manifest.json"
        self.manifest = ProductionManifest(manifest_path)

        # Reference agent for loading character/object refs
        if reference_agent:
            self.reference_agent = reference_agent
        else:
            stem = self.script_path.stem.replace("_full_script", "").replace("_test_page", "")
            assets_path = self.script_path.parent / f"{stem}_assets.json"
            self.reference_agent = OpenAIReferenceAgent(
                assets_path=str(assets_path) if assets_path.exists() else None,
                base_output_dir=self.base_dir,
                style_prompt=style_prompt,
            )

        # Expose reference directories for backward compatibility
        self.char_base_dir = self.reference_agent.char_base_dir
        self.obj_base_dir = self.reference_agent.obj_base_dir
        self.known_objects = self.reference_agent.known_objects
        self.character_map = self.reference_agent.character_map

        # Era constraints (set by caller)
        self.era_constraints = ""

        # Rate limiter for sync mode
        self.sync_limiter = RateLimiter(rpm_limit=config.openai_sync_rpm)

        # Batch / file managers
        self.batch_manager = OpenAIBatchManager(
            poll_interval=config.openai_batch_poll_interval,
            timeout=config.openai_batch_timeout,
        )
        self.file_manager = OpenAIFileManager(
            mapping_path=self.base_dir / "openai_file_mapping.json"
        )

        # Load character arcs data if available
        self.character_arcs = {}
        self.scene_states = {}
        self._load_character_arcs()

    def _load_character_arcs(self):
        """Load character arcs from the enrichment pipeline output."""
        stem = self.script_path.stem.replace("_full_script", "").replace("_test_page", "")
        arcs_path = self.script_path.parent / f"{stem}_character_arcs.json"

        if arcs_path.exists():
            with open(arcs_path, "r") as f:
                self.character_arcs = json.load(f)
            if "scene_states" in self.character_arcs:
                for state in self.character_arcs["scene_states"]:
                    page_num = state.get("page_number", 0)
                    self.scene_states[page_num] = state
            print(f"   Loaded character arcs: {len(self.character_arcs.get('characters', []))} characters")

    # ------------------------------------------------------------------ #
    #  Prompt building
    # ------------------------------------------------------------------ #

    def _build_panel_prompt(
        self,
        page_num: int,
        panel_data: dict,
        prev_context: str = None,
        next_context: str = None,
    ) -> str:
        """Build the image generation prompt for a single panel."""
        # Bubble position -> composition instruction
        bubble_pos = panel_data.get("bubble_position", "top-left")
        comp_map = {
            "top-left": "TOP-LEFT",
            "top-right": "TOP-RIGHT",
            "bottom-left": "BOTTOM-LEFT",
            "bottom-right": "BOTTOM-RIGHT",
        }
        corner = comp_map.get(bubble_pos, "TOP-LEFT")
        composition = (
            f"COMPOSITION: Keep {corner} area uncluttered (simple background) "
            f"for text overlay. Use rule-of-thirds framing."
        )

        # Advice
        advice_data = panel_data.get("advice", {})
        if isinstance(advice_data, dict):
            advice_str = (
                f"Continuity: {advice_data.get('continuity_notes', 'N/A')}. "
                f"Historical: {advice_data.get('historical_constraints', 'N/A')}. "
                f"Gear: {advice_data.get('character_gear', 'N/A')}."
            )
        else:
            advice_str = str(advice_data)

        # Narrative flow
        narrative = ""
        if prev_context:
            narrative += f" Previous panel: {prev_context}."
        if next_context:
            narrative += f" Next panel: {next_context}."

        # Era constraints
        era_block = ""
        if self.era_constraints:
            era_block = f"MANDATORY ERA CONSTRAINTS: {self.era_constraints} "

        # Scene context from character arcs
        scene_ctx = self._get_scene_context(page_num, panel_data)

        # Character descriptions for prompt enrichment
        char_descs = []
        for char_name in panel_data.get("characters", []):
            _, meta = self.reference_agent.load_character_refs(char_name)
            if meta:
                desc = meta.get("description", "")
                if desc:
                    char_descs.append(f"{char_name}: {desc}")

        chars_block = ""
        if char_descs:
            chars_block = "Characters in scene: " + "; ".join(char_descs) + ". "

        prompt = (
            f"Comic panel art. Style: {self.style_prompt}. "
            f"{era_block}"
            f"{scene_ctx} "
            f"Scene: {panel_data['visual_description']}. "
            f"{chars_block}"
            f"{narrative} "
            f"Guidance: {advice_str} "
            f"{composition} "
            f"Requirements: High quality comic panel, era-appropriate clothing and technology. "
            f"CRITICAL: Do NOT render any text, speech bubbles, captions, or empty bounding boxes."
        )

        return prompt.strip()

    def _get_scene_context(self, page_num: int, panel_data: dict) -> str:
        """Build scene context from character arcs."""
        lines = []
        scene_state = self.scene_states.get(page_num, {})
        char_states = scene_state.get("characters", {})

        for char_name in panel_data.get("characters", []):
            if char_name in char_states:
                state = char_states[char_name]
                emotional = state.get("emotional_state", "")
                gear = state.get("gear", [])
                if emotional:
                    lines.append(f"{char_name} emotion: {emotional}")
                if gear:
                    lines.append(f"{char_name} gear: {', '.join(gear)}")

        return ". ".join(lines) + "." if lines else ""

    # ------------------------------------------------------------------ #
    #  Reference agent delegation
    # ------------------------------------------------------------------ #

    async def generate_all_references(self, style: str):
        """Delegate to the OpenAI reference agent."""
        await self.reference_agent.generate_all_references(style)
        self.known_objects = self.reference_agent.known_objects
        self.character_map = self.reference_agent.character_map

    def load_character_refs(self, char_name: str):
        return self.reference_agent.load_character_refs(char_name)

    def load_object_refs(self, obj_name: str):
        return self.reference_agent.load_object_refs(obj_name)

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    async def run_production(self):
        """Dispatch to batch or sync mode based on config."""
        if config.openai_batch_enabled:
            await self._run_batch_production()
        else:
            await self._run_sync_production()

    # ------------------------------------------------------------------ #
    #  Batch mode
    # ------------------------------------------------------------------ #

    async def _run_batch_production(self):
        """Submit all pending panels as batch job(s)."""
        with open(self.script_path, "r") as f:
            script_data = json.load(f)

        # Build batch requests for pending panels
        # Note: Batch mode uses /v1/images/generations (prompt-only) because the
        # /v1/images/edits endpoint doesn't support file references in JSONL format.
        # Character descriptions are embedded in prompts for consistency.
        requests: List[BatchRequest] = []
        panel_metadata: Dict[str, dict] = {}  # custom_id -> {page_num, panel_id}

        for page in script_data:
            page_num = page["page_number"]
            for panel in page["panels"]:
                panel_id = panel["panel_id"]

                if self.manifest.is_panel_complete(page_num, panel_id):
                    continue

                custom_id = f"p{page_num}_panel{panel_id}"
                prompt = self._build_panel_prompt(page_num, panel)

                # Batch mode: always use /v1/images/generations with prompt-only.
                # Reference images via /v1/images/edits require binary uploads
                # which aren't supported in JSONL batch format. Character descriptions
                # are already included in the prompt by _build_panel_prompt().
                # Sync mode (fallback + _run_sync_production) still uses /v1/images/edits
                # with actual image bytes for better consistency.
                body = {
                    "model": config.openai_image_model_primary,
                    "prompt": prompt,
                    "size": config.openai_panel_size,
                    "quality": config.openai_image_quality,
                    "n": 1,
                }
                url = "/v1/images/generations"

                requests.append(BatchRequest(
                    custom_id=custom_id,
                    url=url,
                    body=body,
                ))
                panel_metadata[custom_id] = {
                    "page_num": page_num,
                    "panel_id": panel_id,
                }

        if not requests:
            print("   All panels already generated, nothing to batch.")
            return

        total_panels = sum(len(p["panels"]) for p in script_data)
        print(f"\n[OpenAI Batch] Submitting {len(requests)} panels ({total_panels} total, {total_panels - len(requests)} already done)...")

        # 3. Run batch
        results = await self.batch_manager.run_batch(requests)

        # 4. Save results
        failed_panels = []
        for custom_id, result in results.items():
            meta = panel_metadata.get(custom_id)
            if not meta:
                continue

            page_num = meta["page_num"]
            panel_id = meta["panel_id"]

            if result.success:
                self._save_panel(page_num, panel_id, result.b64_image)
            else:
                print(f"   Panel p{page_num}_panel{panel_id} failed: {result.error}")
                failed_panels.append(meta)

        # 5. Retry failed panels synchronously with fallback models
        if failed_panels:
            print(f"\n[OpenAI] Retrying {len(failed_panels)} failed panels synchronously...")
            for meta in failed_panels:
                page_num = meta["page_num"]
                panel_id = meta["panel_id"]
                # Find the panel data
                for page in script_data:
                    if page["page_number"] == page_num:
                        for panel in page["panels"]:
                            if panel["panel_id"] == panel_id:
                                try:
                                    await self.generate_panel_sync(
                                        page_num, panel,
                                        use_fallback=True,
                                    )
                                except Exception as e:
                                    print(f"   Retry failed for p{page_num}_panel{panel_id}: {e}")
                                break
                        break

        # 6. Optional cleanup of uploaded files
        # Uncomment if you want to free OpenAI storage after each run:
        # await self.file_manager.cleanup_files()

        completed = sum(1 for r in results.values() if r.success) + \
                    sum(1 for m in failed_panels if self.manifest.is_panel_complete(m["page_num"], m["panel_id"]))
        print(f"\n[OpenAI Batch] Production complete: {completed}/{len(requests)} panels generated")

    # ------------------------------------------------------------------ #
    #  Sync mode
    # ------------------------------------------------------------------ #

    async def _run_sync_production(self):
        """Process panels sequentially with rate limiting."""
        with open(self.script_path, "r") as f:
            script_data = json.load(f)

        all_panels = []
        for page in script_data:
            page_num = page["page_number"]
            for panel in page["panels"]:
                all_panels.append({"page_num": page_num, "panel_data": panel})

        total = len(all_panels)
        print(f"\n[OpenAI Sync] Generating {total} panels sequentially...")

        last_panel_image = None

        for page in script_data:
            page_num = page["page_number"]
            panels = page["panels"]
            print(f"\n   Page {page_num} ({len(panels)} panels)...")

            for panel_idx, panel in enumerate(panels):
                panel_id = panel["panel_id"]

                if self.manifest.is_panel_complete(page_num, panel_id):
                    print(f"   Skipping p{page_num}_panel{panel_id} (already done)")
                    continue

                # Build context
                global_idx = sum(
                    len(p["panels"]) for p in script_data if p["page_number"] < page_num
                ) + panel_idx

                prev_ctx = None
                if global_idx > 0:
                    prev = all_panels[global_idx - 1]
                    prev_ctx = prev["panel_data"].get("visual_description", "")

                next_ctx = None
                if global_idx < total - 1:
                    nxt = all_panels[global_idx + 1]
                    next_ctx = nxt["panel_data"].get("visual_description", "")

                try:
                    result = await self.generate_panel_sync(
                        page_num, panel,
                        prev_context=prev_ctx,
                        next_context=next_ctx,
                    )
                    if result is not None:
                        last_panel_image = result
                except Exception as e:
                    print(f"   Panel p{page_num}_{panel_id} failed: {e}")

        print(f"\n[OpenAI Sync] Production complete!")

    async def generate_panel_sync(
        self,
        page_num: int,
        panel_data: dict,
        prev_context: str = None,
        next_context: str = None,
        use_fallback: bool = False,
    ) -> Optional[Image.Image]:
        """
        Generate a single panel using the sync API.

        Args:
            page_num: Page number.
            panel_data: Panel data dict from the script.
            prev_context: Previous panel visual description for context.
            next_context: Next panel visual description for context.
            use_fallback: If True, start from fallback model.

        Returns:
            PIL Image of the generated panel, or None on failure.
        """
        panel_id = panel_data["panel_id"]

        if self.manifest.is_panel_complete(page_num, panel_id):
            return None

        async with self.sync_limiter:
            prompt = self._build_panel_prompt(
                page_num, panel_data, prev_context, next_context
            )

            client = get_openai_client()

            models = [
                config.openai_image_model_primary,
                config.openai_image_model_fallback,
                config.openai_image_model_last_resort,
            ]
            if use_fallback:
                models = models[1:]  # Skip primary

            # Gather reference images for edits endpoint
            ref_images_bytes = []
            for char_name in panel_data.get("characters", []):
                refs, _ = self.reference_agent.load_character_refs(char_name)
                if refs:
                    buf = io.BytesIO()
                    refs[0].save(buf, format="PNG")
                    ref_images_bytes.append(buf.getvalue())

            for obj_name in panel_data.get("key_objects", []):
                folder_name = obj_name.lower().replace(" ", "_")
                refs, _ = self.reference_agent.load_object_refs(folder_name)
                if refs:
                    buf = io.BytesIO()
                    refs[0].save(buf, format="PNG")
                    ref_images_bytes.append(buf.getvalue())

            response = None
            for model in models:
                try:
                    print(f"   [OpenAI] p{page_num}_panel{panel_id} via {model}...")

                    if ref_images_bytes:
                        # Use images.edit with reference images
                        image_inputs = [
                            ("ref.png", io.BytesIO(rb), "image/png")
                            for rb in ref_images_bytes
                        ]
                        response = await client.images.edit(
                            model=model,
                            image=image_inputs,
                            prompt=prompt,
                            size=config.openai_panel_size,
                            quality=config.openai_image_quality,
                            n=1,
                        )
                    else:
                        response = await client.images.generate(
                            model=model,
                            prompt=prompt,
                            size=config.openai_panel_size,
                            quality=config.openai_image_quality,
                            n=1,
                        )
                    break
                except Exception as e:
                    print(f"   Model {model} failed: {e}")
                    continue

            if response is None or not response.data:
                print(f"   All models failed for p{page_num}_panel{panel_id}")
                return None

            b64_data = response.data[0].b64_json
            pil_img = self._save_panel(page_num, panel_id, b64_data)
            return pil_img

    # ------------------------------------------------------------------ #
    #  Saving
    # ------------------------------------------------------------------ #

    def _save_panel(self, page_num: int, panel_id: int, b64_data: str) -> Image.Image:
        """Decode base64 image, save to disk, and update manifest."""
        page_dir = self.output_base_dir / f"page_{page_num}"
        page_dir.mkdir(exist_ok=True)

        img_bytes = base64.b64decode(b64_data)
        pil_img = Image.open(io.BytesIO(img_bytes))

        output_path = page_dir / f"panel_{panel_id}.png"
        pil_img.save(output_path, format="PNG", optimize=True)

        self.manifest.mark_panel_complete(page_num, panel_id)
        print(f"   Saved: {output_path}")

        return pil_img
