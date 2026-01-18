"""Generation page for creating new graphic novels."""
from pathlib import Path

import streamlit as st

from constants import ART_STYLES
from utils import calculate_page_count
from config import ERA_CONSTRAINTS
from ui.session import is_logged_in, navigate_to, render_feedback_link, run_async
from ui.db_operations import db_create_novel, db_update_novel
from ui.pipeline import execute_pipeline


def render_generate_page():
    """Generation page."""
    st.title("📚 Illustrative AI")
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

        content = uploaded_file.getvalue().decode('utf-8')
        word_count = len(content.split())
        st.success(f"Loaded: {uploaded_file.name} ({word_count:,} words)")

        st.subheader("2. Configuration")
        page_mode = st.radio("Pages", ["Quick Preview (10)", "Auto"], horizontal=True)

        if page_mode.startswith("Auto"):
            page_calc = calculate_page_count(word_count)
            st.info(f"Recommended: {page_calc['recommended']} pages")

        style = st.selectbox("Art Style", ART_STYLES)

        # Era/Historical Period Constraints
        st.subheader("3. Historical Era (Optional)")
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

        if st.button("🚀 Generate", type="primary", disabled=st.session_state.is_running, use_container_width=True):
            st.session_state.is_running = True

            target_pages = 10 if page_mode.startswith("Quick") else page_calc['recommended']
            test_mode = page_mode.startswith("Quick")
            title = Path(input_path).stem.replace("-", " ").replace("_", " ").title()

            novel_id = None
            if is_logged_in():
                novel_id = db_create_novel(st.session_state.user_id, title, uploaded_file.name, style, None, target_pages)

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
