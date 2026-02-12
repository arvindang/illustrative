"""Generation page for creating new graphic novels."""
import hashlib
from pathlib import Path

import streamlit as st

from constants import ART_STYLES
from utils import calculate_page_count, get_client
from cost_calculator import (
    estimate_total_cost,
    quick_estimate,
    estimate_characters_from_word_count,
    estimate_objects_from_word_count,
    PRESCAN_COST,
)
from config import ERA_CONSTRAINTS
from ui.session import is_logged_in, navigate_to, render_feedback_link, run_async
from ui.db_operations import db_create_novel, db_update_novel
from ui.pipeline import execute_pipeline


async def run_prescan(content: str, max_chars: int = 50000) -> dict:
    """
    Run AI prescan to extract characters and objects from document.

    Args:
        content: Document text content
        max_chars: Maximum characters to analyze (saves tokens)

    Returns:
        Dict with character_names, object_names lists
    """
    text = content[:max_chars]

    prompt = f"""Analyze this text excerpt and extract:
1. CHARACTER_NAMES: List all named characters (people with proper names, not generic descriptions like "the sailor")
2. KEY_OBJECTS: List important recurring objects, vehicles, or locations that appear multiple times and would need visual consistency

Text (first {len(text)} characters):
---
{text}
---

Respond in this exact format:
CHARACTERS: Name1, Name2, Name3
OBJECTS: Object1, Object2, Object3

If none found, write "CHARACTERS: None" or "OBJECTS: None"."""

    try:
        client = get_client()
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        # Parse response
        result_text = response.text.strip()
        characters = []
        objects = []

        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith('CHARACTERS:'):
                chars_str = line.replace('CHARACTERS:', '').strip()
                if chars_str.lower() != 'none':
                    characters = [c.strip() for c in chars_str.split(',') if c.strip()]
            elif line.startswith('OBJECTS:'):
                objs_str = line.replace('OBJECTS:', '').strip()
                if objs_str.lower() != 'none':
                    objects = [o.strip() for o in objs_str.split(',') if o.strip()]

        return {
            'characters': characters,
            'objects': objects,
            'success': True,
        }

    except Exception as e:
        return {
            'characters': [],
            'objects': [],
            'success': False,
            'error': str(e),
        }


def get_file_hash(content: str) -> str:
    """Generate a hash of file content to detect changes."""
    return hashlib.md5(content.encode()).hexdigest()[:16]


def reset_prescan_state():
    """Reset prescan state when file changes."""
    st.session_state.prescan_complete = False
    st.session_state.prescan_running = False
    st.session_state.prescan_characters = []
    st.session_state.prescan_objects = []
    st.session_state.prescan_file_hash = None


def render_generate_page():
    """Generation page."""
    st.title("Illustrative AI")
    st.caption("Transform literature into graphic novels")

    if is_logged_in():
        if st.button("← Dashboard"):
            navigate_to('dashboard')

    st.subheader("1. Upload Manuscript")
    uploaded_file = st.file_uploader("Choose a .txt file (max 10MB)", type=["txt"])

    if uploaded_file:
        input_path = Path("assets/input") / uploaded_file.name
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        content = uploaded_file.getvalue().decode('utf-8-sig', errors='replace')
        word_count = len(content.split())
        file_hash = get_file_hash(content)

        # Check if file changed - reset prescan if so
        if st.session_state.prescan_file_hash != file_hash:
            reset_prescan_state()
            st.session_state.prescan_file_hash = file_hash

        st.success(f"Loaded: {uploaded_file.name} ({word_count:,} words)")

        # Calculate page recommendations
        page_calc = calculate_page_count(word_count)

        # ================================================================
        # PRESCAN SECTION - AI-powered character/object extraction
        # ================================================================
        st.subheader("2. Document Analysis")

        if not st.session_state.prescan_complete and not st.session_state.prescan_running:
            st.info("Analyze your document to detect characters and objects for accurate cost estimation.")

            if st.button("🔍 Analyze Document", type="secondary", use_container_width=True):
                st.session_state.prescan_running = True
                st.rerun()

        elif st.session_state.prescan_running:
            with st.spinner("Analyzing document with AI..."):
                result = run_async(run_prescan(content))

                if result['success']:
                    st.session_state.prescan_characters = result['characters']
                    st.session_state.prescan_objects = result['objects']
                    st.session_state.prescan_complete = True
                else:
                    # Fallback to heuristics on error
                    st.warning(f"AI analysis failed: {result.get('error', 'Unknown error')}. Using estimates.")
                    st.session_state.prescan_characters = [
                        f"Character {i+1}" for i in range(estimate_characters_from_word_count(word_count))
                    ]
                    st.session_state.prescan_objects = [
                        f"Object {i+1}" for i in range(estimate_objects_from_word_count(word_count))
                    ]
                    st.session_state.prescan_complete = True

                st.session_state.prescan_running = False
                st.rerun()

        # Display prescan results if complete
        if st.session_state.prescan_complete:
            characters = st.session_state.prescan_characters
            objects = st.session_state.prescan_objects

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Characters Detected:** {len(characters)}")
                if characters:
                    with st.expander("View characters"):
                        for char in characters:
                            st.write(f"• {char}")
            with col2:
                st.markdown(f"**Objects Detected:** {len(objects)}")
                if objects:
                    with st.expander("View objects"):
                        for obj in objects:
                            st.write(f"• {obj}")

            # Option to re-analyze
            if st.button("🔄 Re-analyze", type="secondary"):
                reset_prescan_state()
                st.rerun()

        st.divider()

        # ================================================================
        # CONFIGURATION SECTION
        # ================================================================
        st.subheader("3. Configuration")
        page_mode = st.radio("Pages", ["Quick Preview (10)", "Auto"], horizontal=True)

        if page_mode.startswith("Auto"):
            st.info(f"Recommended: {page_calc['recommended']} pages (based on {word_count:,} words)")

        style = st.selectbox("Art Style", ART_STYLES)

        # Era/Historical Period Constraints
        st.subheader("4. Historical Era (Optional)")
        st.caption("Select an era to ensure period-accurate costumes, technology, and props")
        era_options = list(ERA_CONSTRAINTS.keys())
        selected_era = st.selectbox("Era Preset", ["None (auto-detect from text)"] + era_options)

        # Custom era text area
        context_constraints = ""
        if selected_era == "Custom (Enter Below)":
            context_constraints = st.text_area(
                "Custom Era Constraints",
                placeholder="e.g., Setting: 1890s London. CLOTHING: Victorian era dress. TECHNOLOGY: Gas lamps, horse-drawn carriages...",
                height=150
            )
        elif selected_era != "None (auto-detect from text)":
            context_constraints = ERA_CONSTRAINTS.get(selected_era, "")
            with st.expander("View era constraints"):
                st.text(context_constraints)

        st.divider()

        # ================================================================
        # COST ESTIMATION SECTION
        # ================================================================
        st.subheader("5. Cost Estimate")

        # Determine page count for estimation
        if page_mode.startswith("Quick"):
            est_pages = 10
        else:
            est_pages = page_calc['recommended']

        # Use prescan data if available, otherwise use heuristics
        if st.session_state.prescan_complete:
            num_characters = len(st.session_state.prescan_characters)
            num_objects = len(st.session_state.prescan_objects)
            estimate_source = "AI Analysis"
        else:
            num_characters = estimate_characters_from_word_count(word_count)
            num_objects = estimate_objects_from_word_count(word_count)
            estimate_source = "Estimated"

        # Calculate cost estimate
        cost_estimate = estimate_total_cost(
            word_count=word_count,
            page_count=est_pages,
            num_characters=num_characters,
            num_objects=num_objects,
            margin_percent=0.10,
            include_prescan=st.session_state.prescan_complete,
        )
        estimated_cost = cost_estimate.total_cost

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pages", f"{cost_estimate.pages}")
        with col2:
            st.metric("Panels", f"~{cost_estimate.panels}")
        with col3:
            st.metric("Characters", f"{num_characters}", help=estimate_source)
        with col4:
            st.metric("Total Cost", f"${estimated_cost:.2f}")

        # Show estimate accuracy indicator
        if st.session_state.prescan_complete:
            st.caption("✅ Cost based on AI document analysis")
        else:
            st.caption("⚠️ Cost estimated from document length. Run 'Analyze Document' for accuracy.")

        with st.expander("View detailed breakdown"):
            st.markdown(f"""
### Cost Breakdown

| Component | Cost |
|-----------|------|
| Image Generation ({cost_estimate.panels} panels) | ${cost_estimate.breakdown.image_generation:.2f} |
| Character References ({num_characters} × 3 candidates) | ${cost_estimate.breakdown.character_refs:.2f} |
| Object References ({num_objects}) | ${cost_estimate.breakdown.object_refs:.2f} |
| Scripting (7-pass pipeline) | ${cost_estimate.breakdown.scripting_input + cost_estimate.breakdown.scripting_output:.4f} |
| Composition Analysis | ${cost_estimate.breakdown.composition_analysis:.4f} |
| Document Analysis | ${cost_estimate.breakdown.prescan:.2f} |

---

| | |
|---|---|
| **Subtotal (Vertex AI)** | **${cost_estimate.subtotal_vertex:.2f}** |
| Margin ({cost_estimate.margin_percent*100:.0f}%) | ${cost_estimate.margin_fee:.2f} |
| **Total** | **${cost_estimate.total_cost:.2f}** |

_Context caching saves ${cost_estimate.breakdown.cache_savings:.4f} on repeated content_
            """)

        st.divider()

        # ================================================================
        # GENERATE BUTTON
        # ================================================================
        # Disable generate if prescan is running
        generate_disabled = st.session_state.is_running or st.session_state.prescan_running

        if st.button("🚀 Generate Graphic Novel", type="primary", disabled=generate_disabled, use_container_width=True):
            st.session_state.is_running = True

            target_pages = 10 if page_mode.startswith("Quick") else page_calc['recommended']
            test_mode = page_mode.startswith("Quick")
            title = Path(input_path).stem.replace("-", " ").replace("_", " ").title()

            novel_id = None
            if is_logged_in():
                novel_id = db_create_novel(
                    st.session_state.user_id, title, uploaded_file.name,
                    style, None, target_pages, estimated_cost=estimated_cost
                )

            with st.status("🚀 Running...", expanded=True) as status:
                try:
                    run_async(execute_pipeline(status, str(input_path), style, target_pages, test_mode, novel_id, context_constraints))
                    status.update(label="✅ Complete!", state="complete")
                    st.session_state.pipeline_complete = True
                    if novel_id:
                        db_update_novel(novel_id, status="completed",
                                       pdf_storage_key=st.session_state.output_paths.get('pdf_storage_key'),
                                       epub_storage_key=st.session_state.output_paths.get('epub_storage_key'))
                except Exception as e:
                    status.update(label=f"❌ Error", state="error")
                    error_msg = str(e)[:2000]  # Truncate to fit column
                    st.error(error_msg)
                    if novel_id:
                        db_update_novel(novel_id, status="failed", error_message=error_msg)
                finally:
                    st.session_state.is_running = False

    # ================================================================
    # DOWNLOAD SECTION (after generation complete)
    # ================================================================
    if st.session_state.pipeline_complete and st.session_state.output_paths:
        st.divider()
        st.subheader("📦 Download")
        pdf_path = st.session_state.output_paths.get('pdf_path')
        epub_path = st.session_state.output_paths.get('epub_path')
        input_stem = st.session_state.output_paths.get('input_stem', 'novel')

        col1, col2 = st.columns(2)
        with col1:
            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 PDF", f.read(), f"{input_stem}.pdf", "application/pdf", use_container_width=True)
        with col2:
            if epub_path and Path(epub_path).exists():
                with open(epub_path, "rb") as f:
                    st.download_button("📚 EPUB", f.read(), f"{input_stem}.epub", "application/epub+zip", use_container_width=True)

    # Feedback link for logged-in users
    if is_logged_in():
        st.divider()
        render_feedback_link()
