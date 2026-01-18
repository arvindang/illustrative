"""
ScriptingAgent - Main orchestration class for the graphic novel scripting pipeline.

This agent coordinates an 8-pass enrichment pipeline:
- Pass 0: Global Context (provided via context_constraints)
- Pass 1: Beat Analysis
- Pass 1.5: Adaptation Filter
- Pass 2: Director Pass (Blueprint)
- Pass 3: Character Deep Dive
- Pass 4: Asset Manifest
- Pass 5: Scriptwriter Pass
- Pass 5.5: Dialogue Polish
- Pass 6: Validation + Auto-Fix
"""
import json
import hashlib
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing config
load_dotenv()

from google.genai import types

from utils import (
    get_tpm_limiter, estimate_tokens_for_cache, get_client
)
from config import config

# Import pass functions
from agents.scripting.passes import (
    analyze_narrative_beats,
    generate_adaptation_filter,
    generate_pacing_blueprint,
    generate_character_deep_dive,
    generate_asset_manifest,
    write_page_script,
    polish_dialogue,
)


class ScriptingAgent:
    """
    Main orchestration class for the graphic novel scripting pipeline.

    Usage:
        agent = ScriptingAgent("path/to/book.txt")
        script = await agent.generate_script(style="Lush Watercolor", test_mode=True)
    """

    def __init__(self, book_path: str, base_output_dir: Path = None):
        """
        Initialize the ScriptingAgent.

        Args:
            book_path: Path to the source text file
            base_output_dir: Output directory for generated files (defaults to config.output_dir)
        """
        self.book_path = Path(book_path)
        self.output_dir = Path(base_output_dir) if base_output_dir else config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_name = None

    def load_content(self, test_mode=True) -> str:
        """
        Loads the book content.

        Args:
            test_mode: If True, returns a truncated version for testing

        Returns:
            Book content as string
        """
        with open(self.book_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            if test_mode:
                # Take a generous slice for testing
                return full_text[:50000]
            return full_text

    async def _get_or_create_cache(self, content: str) -> str:
        """
        Uploads the book content to Gemini's Context Cache.

        Args:
            content: Full book content

        Returns:
            Cache name to reference in subsequent calls, or None if caching failed
        """
        # Create a deterministic hash for the cache name based on content
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        print(f"💾 Caching book content ({len(content)} chars) to Gemini Context Cache...")

        # Acquire TPM capacity for cache creation
        estimated_tokens = estimate_tokens_for_cache(content)
        await get_tpm_limiter().acquire(estimated_tokens)

        try:
            # Using a distinct display name helps debugging
            display_name = f"book_cache_{self.book_path.stem}_{content_hash[:6]}"

            cached_content = await get_client().aio.caches.create(
                model=config.scripting_model_global_context,
                config=types.CreateCachedContentConfig(
                    display_name=display_name,
                    contents=[content],
                    ttl="3600s"  # 1 hour TTL
                )
            )

            self.cache_name = cached_content.name
            print(f"✅ Cache created: {self.cache_name} (TTL: 1h)")
            return self.cache_name

        except Exception as e:
            print(f"⚠️ Cache creation failed: {e}")
            print("   Falling back to sending full text in context (slower/more expensive).")
            return None

    def normalize_character_name(self, name: str, registry: dict) -> tuple:
        """
        Normalizes character names to handle variants and duplicates.

        Args:
            name: Character name to normalize
            registry: Dict mapping canonical names to sets of variants

        Returns:
            Tuple of (canonical_name, folder_safe_name)
        """
        if not name:
            return None, None

        titles = ["captain", "professor", "doctor", "dr", "mr", "mrs", "miss", "sir", "lord", "lady"]
        parts = name.strip().split()
        base_name = parts[-1].lower() if parts else ""

        # Check if this base name already has a canonical form
        for canonical, variants in registry.items():
            canonical_base = canonical.split()[-1].lower()
            if base_name == canonical_base:
                registry[canonical].add(name)
                folder_name = canonical.lower().replace(" ", "_")
                return canonical, folder_name

        # New character - prefer names with titles
        has_title = any(part.lower() in titles for part in parts[:-1])
        canonical = name if has_title or len(parts) > 1 else name
        registry[canonical] = {name}
        folder_name = canonical.lower().replace(" ", "_")
        return canonical, folder_name

    def extract_characters_from_blueprint(self, blueprint: list) -> list:
        """
        Extracts unique, normalized character names from the blueprint.

        Args:
            blueprint: List of page blueprint dictionaries

        Returns:
            List of canonical character names
        """
        registry = {}
        canonical_chars = []

        for page in blueprint:
            for char_name in page.get('key_characters', []):
                canonical, _ = self.normalize_character_name(char_name, registry)
                if canonical and canonical not in canonical_chars:
                    canonical_chars.append(canonical)

        return canonical_chars

    async def generate_script(
        self,
        style: str,
        test_mode: bool = True,
        context_constraints: str = "",
        target_page_override: int = None
    ) -> list:
        """
        Main orchestration method - 8-pass enrichment pipeline.

        PASS 0: Global Context (provided via context_constraints)
        PASS 1: Beat Analysis - Extract narrative beats with visual potential scores
        PASS 1.5: Adaptation Filter - Identify essential/condensable/cuttable content
        PASS 2: Director Pass - Blueprint with spread awareness + cliffhangers
        PASS 3: Character Deep Dive - Arcs, voice profiles, scene-specific gear
        PASS 4: Asset Manifest - Characters, objects, locations, interaction rules
        PASS 5: Scriptwriter Pass - Hybrid sequential/parallel generation
        PASS 5.5: Dialogue Polish - Cross-page dialogue flow improvements
        PASS 6: Validation + Auto-Fix - Validate and fix common issues

        Args:
            style: Art style for the graphic novel
            test_mode: If True, generates a shorter test version
            context_constraints: Era/setting constraints
            target_page_override: Override the default page count

        Returns:
            List of page script dictionaries
        """
        from validators import validate_and_autofix_script

        # Output paths
        suffix = "_test_page.json" if test_mode else "_full_script.json"
        output_file = self.output_dir / f"{self.book_path.stem}{suffix}"

        if output_file.exists() and not test_mode:
            print(f"⏭️  Skipping Script Generation (Already exists: {output_file})")
            with open(output_file, "r") as f:
                return json.load(f)

        # Load Content
        full_text = self.load_content(test_mode=False)  # Always load full for caching
        target_pages = target_page_override or (3 if test_mode else 10)

        print(f"\n📚 Scripting '{self.book_path.stem}'")
        print(f"   Target Length: {target_pages} pages")
        print(f"   Style: {style}")

        # Create Cache (The Enabler)
        cache_name = await self._get_or_create_cache(full_text)

        # Log context constraints if provided
        if context_constraints:
            print(f"   Era/Context: {context_constraints[:100]}...")

        # ========== PASS 1: Beat Analysis ==========
        print("\n🎭 PASS 1: Analyzing narrative beats...")
        beat_map = await analyze_narrative_beats(cache_name, full_text, target_pages)

        # Save beat map
        beats_path = self.output_dir / f"{self.book_path.stem}_beats.json"
        with open(beats_path, "w") as f:
            json.dump(beat_map, f, indent=2)
        print(f"✅ Beat analysis complete: {beats_path}")
        print(f"   Found {len(beat_map.get('beats', []))} narrative beats")

        # ========== PASS 1.5: Adaptation Filter ==========
        print("\n✂️  PASS 1.5: Running adaptation filter...")
        adaptation_filter = await generate_adaptation_filter(
            cache_name, full_text, beat_map, target_pages
        )

        # Save adaptation filter
        filter_path = self.output_dir / f"{self.book_path.stem}_adaptation.json"
        with open(filter_path, "w") as f:
            json.dump(adaptation_filter, f, indent=2)
        print(f"✅ Adaptation filter complete: {filter_path}")

        # ========== PASS 2: Director Pass (Blueprint) ==========
        print("\n🎬 PASS 2: Generating pacing blueprint...")
        blueprint = await generate_pacing_blueprint(
            cache_name, full_text, target_pages, style, context_constraints, beat_map, adaptation_filter
        )

        # Save blueprint
        blueprint_path = self.output_dir / f"{self.book_path.stem}_blueprint.json"
        with open(blueprint_path, "w") as f:
            json.dump(blueprint, f, indent=2)
        print(f"✅ Blueprint created: {blueprint_path}")

        # ========== PASS 3: Character Deep Dive ==========
        print("\n👥 PASS 3: Generating character deep dive...")
        character_arcs = await generate_character_deep_dive(
            cache_name, full_text, blueprint, self.extract_characters_from_blueprint
        )

        # Save character arcs
        arcs_path = self.output_dir / f"{self.book_path.stem}_character_arcs.json"
        with open(arcs_path, "w") as f:
            json.dump(character_arcs, f, indent=2)
        print(f"✅ Character arcs created: {arcs_path}")
        print(f"   Tracking {len(character_arcs.get('characters', []))} characters")

        # ========== PASS 4: Asset Manifest ==========
        print("\n📋 PASS 4: Generating asset manifest...")
        asset_manifest = await generate_asset_manifest(
            cache_name, full_text, blueprint, style, context_constraints, character_arcs,
            self.extract_characters_from_blueprint
        )

        # Save asset manifest
        manifest_path = self.output_dir / f"{self.book_path.stem}_assets.json"
        with open(manifest_path, "w") as f:
            json.dump(asset_manifest, f, indent=2)
        print(f"✅ Asset manifest created: {manifest_path}")

        # ========== PASS 5: Scriptwriter Pass (Hybrid Sequential/Parallel) ==========
        print(f"\n⚡ PASS 5: Generating panel scripts for {len(blueprint)} pages...")
        print(f"   Using voice profiles from {len(character_arcs.get('characters', []))} characters")

        # Identify dialogue-heavy sequences that need sequential generation
        DIALOGUE_SCENE_TYPES = {'dialogue', 'formal', 'flashback'}

        dialogue_sequences = []  # List of (start_idx, end_idx) for sequential generation
        current_seq_start = None

        for i, item in enumerate(blueprint):
            scene_type = item.get('scene_type', 'dialogue')
            is_dialogue = scene_type in DIALOGUE_SCENE_TYPES

            if is_dialogue:
                if current_seq_start is None:
                    current_seq_start = i
            else:
                if current_seq_start is not None and i - current_seq_start >= 2:
                    # End of a dialogue sequence (2+ pages)
                    dialogue_sequences.append((current_seq_start, i - 1))
                current_seq_start = None

        # Don't forget the last sequence if it ends at the last page
        if current_seq_start is not None and len(blueprint) - current_seq_start >= 2:
            dialogue_sequences.append((current_seq_start, len(blueprint) - 1))

        # Track which pages are in dialogue sequences
        sequential_pages = set()
        for start, end in dialogue_sequences:
            for i in range(start, end + 1):
                sequential_pages.add(i)

        parallel_count = len(blueprint) - len(sequential_pages)
        seq_count = len(sequential_pages)
        print(f"   📝 Sequential generation: {seq_count} pages in {len(dialogue_sequences)} dialogue sequence(s)")
        print(f"   ⚡ Parallel generation: {parallel_count} action/establishing pages")

        # Generate all pages, using sequential for dialogue sequences
        full_script = []
        generated_scripts = {}  # idx -> script for lookups

        # PHASE 1: Generate dialogue sequences SEQUENTIALLY
        for seq_start, seq_end in dialogue_sequences:
            print(f"   🗣️  Generating dialogue sequence pages {seq_start + 1}-{seq_end + 1} sequentially...")
            prev_script = generated_scripts.get(seq_start - 1)  # Previous page if exists

            for i in range(seq_start, seq_end + 1):
                item = blueprint[i]
                prev_page = blueprint[i - 1] if i > 0 else None
                next_page = blueprint[i + 1] if i < len(blueprint) - 1 else None

                try:
                    script = await write_page_script(
                        item, style, context_constraints, character_arcs, asset_manifest,
                        prev_page, next_page, prev_script
                    )
                    full_script.append(script)
                    generated_scripts[i] = script
                    prev_script = script  # Pass to next page in sequence
                    print(f"      ✅ Page {item.get('page_number')} complete")
                except Exception as e:
                    print(f"      ❌ Page {item.get('page_number')} failed: {e}")

        # PHASE 2: Generate remaining pages IN PARALLEL
        parallel_tasks = []
        parallel_indices = []

        for i, item in enumerate(blueprint):
            if i not in sequential_pages:
                prev_page = blueprint[i - 1] if i > 0 else None
                next_page = blueprint[i + 1] if i < len(blueprint) - 1 else None
                prev_script = generated_scripts.get(i - 1)

                parallel_tasks.append(
                    write_page_script(
                        item, style, context_constraints, character_arcs, asset_manifest,
                        prev_page, next_page, prev_script
                    )
                )
                parallel_indices.append(i)

        if parallel_tasks:
            print(f"   ⚡ Generating {len(parallel_tasks)} parallel pages...")
            results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

            for idx, result in zip(parallel_indices, results):
                if isinstance(result, Exception):
                    print(f"   ❌ Page {blueprint[idx].get('page_number')} failed: {result}")
                else:
                    full_script.append(result)
                    generated_scripts[idx] = result

        # Identify failed pages for retry
        failed_pages = []
        failed_blueprints = {}

        for i, item in enumerate(blueprint):
            if i not in generated_scripts:
                page_num = item.get('page_number', i + 1)
                failed_pages.append(page_num)
                failed_blueprints[page_num] = (item, i)

        # RETRY LOGIC: Attempt to regenerate failed pages (up to 2 retries)
        MAX_RETRIES = 2
        retry_count = 0

        while failed_pages and retry_count < MAX_RETRIES:
            retry_count += 1
            print(f"\n🔄 Retry attempt {retry_count}/{MAX_RETRIES} for {len(failed_pages)} failed page(s)...")

            retry_tasks = []
            for page_num in failed_pages:
                item, idx = failed_blueprints[page_num]
                prev_page = blueprint[idx - 1] if idx > 0 else None
                next_page = blueprint[idx + 1] if idx < len(blueprint) - 1 else None
                prev_script = generated_scripts.get(idx - 1)
                retry_tasks.append(
                    write_page_script(
                        item, style, context_constraints, character_arcs, asset_manifest,
                        prev_page, next_page, prev_script
                    )
                )

            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

            still_failed = []
            for i, result in enumerate(retry_results):
                page_num = failed_pages[i]
                item, idx = failed_blueprints[page_num]
                if isinstance(result, Exception):
                    print(f"   ❌ Page {page_num} retry {retry_count} failed: {result}")
                    still_failed.append(page_num)
                else:
                    print(f"   ✅ Page {page_num} succeeded on retry {retry_count}")
                    full_script.append(result)
                    generated_scripts[idx] = result

            failed_pages = still_failed

        if failed_pages:
            print(f"⚠️ {len(failed_pages)} page(s) still failed after {MAX_RETRIES} retries: {failed_pages}")
            print("   These pages will be missing from the final script. Consider re-running the pipeline.")

        # Sort by page number
        full_script.sort(key=lambda x: x['page_number'])

        # ========== PASS 5.5: Dialogue Polish Pass ==========
        full_script = await polish_dialogue(
            full_script,
            character_arcs,
            context_constraints
        )

        # ========== PASS 6: Validation + Auto-Fix ==========
        print("\n🔍 PASS 6: Validating and auto-fixing script...")

        # Extract era from context_constraints for validation
        era = ""
        if context_constraints:
            if "1860" in context_constraints or "Victorian" in context_constraints:
                era = "1860s Victorian"
            elif "1920" in context_constraints or "Art Deco" in context_constraints:
                era = "1920s Art Deco"
            elif "Medieval" in context_constraints or "medieval" in context_constraints:
                era = "Medieval Fantasy"

        fixed_script, validation_report = validate_and_autofix_script(
            script=full_script,
            era=era,
            character_arcs=character_arcs,
            assets=asset_manifest
        )

        # Save validation report
        validation_path = self.output_dir / f"{self.book_path.stem}_validation.json"
        with open(validation_path, "w") as f:
            report_dict = {
                "total_issues": validation_report.total_issues,
                "auto_fixed": validation_report.auto_fixed,
                "warnings": validation_report.warnings,
                "manual_review": validation_report.manual_review,
                "script_modified": validation_report.script_modified,
                "issues": [
                    {
                        "issue_type": issue.issue_type,
                        "page": issue.page,
                        "panel": issue.panel,
                        "severity": issue.severity,
                        "description": issue.description,
                        "fix_applied": issue.fix_applied,
                        "original_value": issue.original_value,
                        "fixed_value": issue.fixed_value
                    }
                    for issue in validation_report.issues
                ]
            }
            json.dump(report_dict, f, indent=2)

        print(f"✅ Validation complete: {validation_path}")
        print(f"   Total issues: {validation_report.total_issues}")
        print(f"   Auto-fixed: {validation_report.auto_fixed}")
        print(f"   Manual review needed: {validation_report.manual_review}")

        # Use fixed script if modifications were made
        if validation_report.script_modified:
            full_script = fixed_script
            print("   Applied auto-fixes to script")

        # Save Final Script
        with open(output_file, "w") as f:
            json.dump(full_script, f, indent=2)

        print(f"\n🎉 Script Generation Complete! Saved to: {output_file}")
        return full_script


if __name__ == "__main__":
    # Test run
    agent = ScriptingAgent("assets/input/20-thousand-leagues-under-the-sea.txt")
    asyncio.run(agent.generate_script(
        style="Lush Watercolor",
        test_mode=True,
        target_page_override=3
    ))
