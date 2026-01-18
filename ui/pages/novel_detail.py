"""Novel detail page with resume functionality."""
from datetime import datetime
from pathlib import Path

import streamlit as st

from ui.session import navigate_to, render_feedback_link, run_async
from ui.db_operations import db_get_novel, db_update_novel, db_delete_novel, get_download_url
from ui.pipeline import resume_pipeline


def get_partial_content(user_id: str, novel_id: str) -> dict:
    """
    Check for partial content on disk for a novel.

    Returns:
        Dict with: has_content, panel_images (list), final_pages (list), has_script
    """
    base_dir = Path("assets/output") / str(user_id) / str(novel_id)

    result = {
        "has_content": False,
        "panel_images": [],
        "final_pages": [],
        "has_script": False,
        "script_path": None
    }

    if not base_dir.exists():
        return result

    # Check for panel images
    pages_dir = base_dir / "pages"
    if pages_dir.exists():
        for page_dir in sorted(pages_dir.iterdir()):
            if page_dir.is_dir():
                for panel in sorted(page_dir.glob("panel_*.png")):
                    result["panel_images"].append(str(panel))

    # Check for final composed pages
    final_dir = base_dir / "final_pages"
    if final_dir.exists():
        result["final_pages"] = sorted([str(p) for p in final_dir.glob("page_*.png")])

    # Check for script
    for script_file in base_dir.glob("*_script.json"):
        result["has_script"] = True
        result["script_path"] = str(script_file)
        break
    for script_file in base_dir.glob("*_full_script.json"):
        result["has_script"] = True
        result["script_path"] = str(script_file)
        break

    result["has_content"] = bool(result["panel_images"] or result["final_pages"])

    return result


def can_resume_novel(novel: dict) -> tuple:
    """
    Check if a novel can be resumed.

    Returns:
        Tuple of (can_resume: bool, resume_stage: str, message: str)
    """
    if novel['status'] == 'completed':
        return False, None, "Already completed"

    manifest_path = novel.get('manifest_path')
    if not manifest_path or not Path(manifest_path).exists():
        # Check if we can find content anyway
        user_id = st.session_state.get('user_id')
        if user_id:
            partial = get_partial_content(user_id, novel['id'])
            if partial['has_script']:
                return True, 'illustrating', f"Script found at {partial['script_path']}"
        return False, None, "No manifest or script found"

    # Check manifest for progress
    from utils import get_manifest_progress
    progress = get_manifest_progress(manifest_path)

    if progress.get('error'):
        return False, None, f"Manifest error: {progress['error']}"

    # Determine resume stage
    base_dir = Path(manifest_path).parent
    script_files = list(base_dir.glob("*_script.json")) + list(base_dir.glob("*_full_script.json"))

    if not script_files:
        return False, None, "No script file found"

    if progress['panels_completed'] > 0:
        return True, 'illustrating', f"{progress['panels_completed']} panels already generated"

    return True, 'illustrating', "Script exists, can resume illustration"


def render_novel_detail_page():
    """Novel detail/show page."""
    novel_id = st.query_params.get("novel_id")

    if not novel_id:
        st.error("No novel specified")
        if st.button("← Back to Dashboard"):
            navigate_to('dashboard')
        return

    novel = db_get_novel(novel_id, st.session_state.user_id)

    if not novel:
        st.error("Novel not found")
        if st.button("← Back to Dashboard"):
            navigate_to('dashboard')
        return

    # Header with back button
    if st.button("← Back to Dashboard"):
        navigate_to('dashboard')

    st.title(novel['title'])

    # Status badge with progress info
    status = novel['status']
    if status == 'completed':
        st.success("Completed")
    elif status == 'processing':
        # Show detailed progress if available
        stage = novel.get('current_stage', 'processing')
        stage_display = {
            'scripting': 'Scripting',
            'illustrating': 'Illustrating',
            'compositing': 'Compositing',
            'exporting': 'Exporting'
        }.get(stage, 'Processing')

        # Check manifest for actual progress
        if novel.get('manifest_path') and Path(novel['manifest_path']).exists():
            from utils import get_manifest_progress
            progress = get_manifest_progress(novel['manifest_path'])
            panels_done = progress.get('panels_completed', 0)
            panels_total = novel.get('panels_total', 0)

            if panels_done > 0:
                if panels_total > 0:
                    pct = panels_done / panels_total
                    st.info(f"{stage_display}... ({panels_done}/{panels_total} panels)")
                    st.progress(pct)
                else:
                    st.info(f"{stage_display}... ({panels_done} panels generated)")
            else:
                st.info(f"{stage_display}...")
        else:
            st.info(f"{stage_display}...")
    else:
        st.error("Failed")

    # Metadata
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Art Style**")
        st.write(novel['art_style'] or "Not specified")
        st.markdown("**Pages**")
        st.write(novel['page_count'] or "Unknown")
    with col2:
        st.markdown("**Source File**")
        st.write(novel['source_filename'] or "Unknown")
        st.markdown("**Created**")
        if novel['created_at']:
            created = datetime.fromisoformat(novel['created_at'].replace('Z', '+00:00'))
            st.write(created.strftime("%B %d, %Y at %I:%M %p"))
        else:
            st.write("Unknown")

    # Error message for failed runs
    if status == 'failed' and novel.get('error_message'):
        st.divider()
        st.markdown("**Error Details**")
        st.error(novel['error_message'])

    # Partial content and resume options for failed/processing novels
    if status in ('failed', 'processing'):
        user_id = st.session_state.get('user_id')
        if user_id:
            partial = get_partial_content(user_id, novel_id)

            # Show partial content if available
            if partial["has_content"]:
                st.divider()
                st.subheader("Partial Content Available")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Panel Images", len(partial["panel_images"]))
                with col2:
                    st.metric("Composed Pages", len(partial["final_pages"]))

                # Show preview of first few final pages
                if partial["final_pages"]:
                    st.write("**Preview of completed pages:**")
                    preview_count = min(4, len(partial["final_pages"]))
                    preview_cols = st.columns(preview_count)
                    for i in range(preview_count):
                        with preview_cols[i]:
                            st.image(partial["final_pages"][i], caption=f"Page {i+1}", use_container_width=True)

            # Resume option
            resumable, resume_stage, resume_msg = can_resume_novel(novel)

            if resumable:
                st.divider()
                st.subheader("Resume Generation")
                st.info(f"This novel can be resumed. {resume_msg}")

                if st.button("Resume Generation", type="primary", use_container_width=True):
                    with st.status("Resuming...", expanded=True) as resume_status:
                        try:
                            db_update_novel(novel_id, status="processing")
                            storage_keys = run_async(resume_pipeline(resume_status, novel))

                            resume_status.update(label="Completed!", state="complete")
                            db_update_novel(
                                novel_id,
                                status="completed",
                                pdf_storage_key=storage_keys.get('pdf_storage_key'),
                                epub_storage_key=storage_keys.get('epub_storage_key')
                            )
                            st.rerun()
                        except Exception as e:
                            resume_status.update(label="Error", state="error")
                            error_msg = str(e)[:2000]
                            st.error(error_msg)
                            db_update_novel(novel_id, status="failed", error_message=error_msg)

    # Downloads (show for completed novels)
    if status == 'completed':
        st.divider()
        st.subheader("Downloads")

        if novel['has_pdf'] or novel['has_epub']:
            col1, col2 = st.columns(2)
            with col1:
                if novel['has_pdf'] and novel.get('pdf_storage_key'):
                    pdf_url = get_download_url(novel['pdf_storage_key'])
                    if pdf_url:
                        st.link_button("Download PDF", pdf_url, use_container_width=True)
                    else:
                        st.button("PDF unavailable", disabled=True, use_container_width=True)
            with col2:
                if novel['has_epub'] and novel.get('epub_storage_key'):
                    epub_url = get_download_url(novel['epub_storage_key'])
                    if epub_url:
                        st.link_button("Download EPUB", epub_url, use_container_width=True)
                    else:
                        st.button("EPUB unavailable", disabled=True, use_container_width=True)
        else:
            st.info("No downloads available. This run completed before cloud storage was configured, or files were not uploaded.")

    # Delete action
    st.divider()
    with st.expander("Danger Zone"):
        if st.button("Delete this novel", type="secondary"):
            db_delete_novel(novel_id, st.session_state.user_id)
            navigate_to('dashboard')

    render_feedback_link()
