"""Pipeline execution logic for graphic novel generation."""
from pathlib import Path

import streamlit as st

from ui.db_operations import db_update_novel


async def execute_pipeline(status, input_path: str, style: str, target_pages: int, test_mode: bool, novel_id: str = None, context_constraints: str = ""):
    """Execute the full pipeline with progress tracking."""
    # Lazy import heavy modules only when actually generating
    from agents import ScriptingAgent, IllustratorAgent, CompositorAgent

    input_stem = Path(input_path).stem

    # Create isolated output directory
    user_id = st.session_state.get('user_id')
    if user_id and novel_id:
        base_output_dir = Path("assets/output") / str(user_id) / str(novel_id)
    elif novel_id:
        base_output_dir = Path("assets/output") / "anonymous" / str(novel_id)
    else:
        # Fallback for temporary runs
        import uuid
        run_id = str(uuid.uuid4())
        base_output_dir = Path("assets/output") / "temp" / run_id

    base_output_dir.mkdir(parents=True, exist_ok=True)
    status.write(f"📂 Output directory: {base_output_dir}")

    # Set up manifest path for progress tracking
    manifest_path = str(base_output_dir / "production_manifest.json")

    # STEP 1: SCRIPTING
    status.write("--- STEP 1/4: SCRIPTING ---")
    if novel_id:
        db_update_novel(novel_id, current_stage='scripting', manifest_path=manifest_path)
    status.write("📚 Loading manuscript...")
    scripter = ScriptingAgent(input_path, base_output_dir=base_output_dir)

    if context_constraints:
        status.write(f"🕰️ Era constraints active: {context_constraints[:80]}...")

    status.write("💾 Caching book content...")
    script = await scripter.generate_script(
        style=style, test_mode=test_mode, context_constraints=context_constraints, target_page_override=target_pages
    )
    status.write(f"✅ Script complete: {len(script)} pages generated")

    suffix = "_test_page.json" if test_mode else "_full_script.json"
    script_path = base_output_dir / f"{input_stem}{suffix}"

    # Calculate total panels for progress tracking
    total_panels = sum(len(page.get('panels', [])) for page in script)

    # STEP 2: ILLUSTRATION
    status.write("")
    status.write("--- STEP 2/4: ILLUSTRATION ---")
    if novel_id:
        db_update_novel(novel_id, current_stage='illustrating', pages_total=len(script),
                       panels_total=total_panels, can_resume=True)

    style_prompt = f"{style} style, high-quality graphic novel art."
    illustrator = IllustratorAgent(str(script_path), style_prompt, base_output_dir=base_output_dir)

    status.write("🎨 Generating character & object reference sheets...")
    await illustrator.generate_all_references(style=style)

    status.write("🖼️ Generating panel images...")
    await illustrator.run_production()
    status.write("✅ Illustration complete!")

    # STEP 3: COMPOSITION
    status.write("")
    status.write("--- STEP 3/4: COMPOSITION ---")
    if novel_id:
        db_update_novel(novel_id, current_stage='compositing')

    status.write("📐 Assembling final pages...")
    compositor = CompositorAgent(str(script_path), base_output_dir=base_output_dir)
    compositor.run()
    status.write("✅ Composition complete!")

    # STEP 4: EXPORT
    status.write("")
    status.write("--- STEP 4/4: EXPORT ---")
    if novel_id:
        db_update_novel(novel_id, current_stage='exporting')

    output_base = base_output_dir / input_stem
    title = input_stem.replace("-", " ").replace("_", " ").title()

    if novel_id:
        status.write("📄 Generating PDF & EPUB and uploading to cloud...")
        storage_keys = compositor.export_and_upload(output_base, novel_id, title=title)
        pdf_path = output_base.with_suffix(".pdf")
        epub_path = output_base.with_suffix(".epub")
    else:
        status.write("📄 Generating PDF...")
        pdf_path = compositor.export_pdf(output_base)
        status.write("📚 Generating EPUB...")
        epub_path = compositor.export_epub(output_base, title=title)
        storage_keys = {}

    status.write("✅ Export complete!")

    st.session_state.output_paths = {
        'script_path': str(script_path), 'output_base': str(output_base), 'input_stem': input_stem,
        'pdf_path': str(pdf_path) if pdf_path else None,
        'epub_path': str(epub_path) if epub_path else None,
        **storage_keys
    }
    return True


async def resume_pipeline(status, novel: dict):
    """Resume a partially completed pipeline from the last checkpoint."""
    # Lazy import heavy modules only when actually generating
    from agents import IllustratorAgent, CompositorAgent

    novel_id = novel['id']
    user_id = st.session_state.user_id

    base_dir = Path("assets/output") / str(user_id) / str(novel_id)

    # Find the script file
    script_files = list(base_dir.glob("*_script.json")) + list(base_dir.glob("*_full_script.json"))
    if not script_files:
        raise Exception("Cannot resume: Script file not found")

    script_path = script_files[0]

    status.write(f"Resuming from: {script_path.name}")

    # Load script to get metadata
    import json
    with open(script_path, 'r') as f:
        script = json.load(f)

    total_panels = sum(len(page.get('panels', [])) for page in script)

    # Check current progress
    from utils import get_manifest_progress
    manifest_path = base_dir / "production_manifest.json"
    if manifest_path.exists():
        progress = get_manifest_progress(str(manifest_path))
        status.write(f"   {progress['panels_completed']} panels already complete")
    else:
        status.write("   Starting fresh illustration...")

    # Update DB with resume info
    db_update_novel(novel_id, current_stage='illustrating', pages_total=len(script),
                   panels_total=total_panels, can_resume=True)

    # Resume illustration (manifest will skip completed panels)
    style_prompt = f"{novel['art_style']} style, high-quality graphic novel art."
    illustrator = IllustratorAgent(str(script_path), style_prompt, base_output_dir=base_dir)

    status.write("--- RESUMING ILLUSTRATION ---")
    status.write("🎨 Checking character references...")
    await illustrator.generate_all_references(style=novel['art_style'])

    status.write("🖼️ Generating remaining panel images...")
    await illustrator.run_production()
    status.write("✅ Illustration complete!")

    # Continue with composition
    status.write("")
    status.write("--- COMPOSITION ---")
    db_update_novel(novel_id, current_stage='compositing')

    compositor = CompositorAgent(str(script_path), base_output_dir=base_dir)
    compositor.run()
    status.write("✅ Composition complete!")

    # Export
    status.write("")
    status.write("--- EXPORT ---")
    db_update_novel(novel_id, current_stage='exporting')

    input_stem = script_path.stem.replace('_full_script', '').replace('_test_page', '')
    output_base = base_dir / input_stem
    title = novel['title']

    status.write("📄 Generating PDF & EPUB and uploading to cloud...")
    storage_keys = compositor.export_and_upload(output_base, novel_id, title=title)
    status.write("✅ Export complete!")

    return storage_keys
