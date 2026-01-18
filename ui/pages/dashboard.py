"""User dashboard page."""
import streamlit as st

from ui.session import navigate_to, render_feedback_link
from ui.db_operations import db_list_novels, db_delete_novel


def render_dashboard():
    """User dashboard."""
    st.title("📚 My Graphic Novels")

    if st.session_state.user_email:
        st.caption(f"Logged in as {st.session_state.user_email}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Novel", use_container_width=True, type="primary"):
            navigate_to('generate')
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.user_email = None
            navigate_to('home')

    st.divider()

    novels = db_list_novels(st.session_state.user_id)
    if not novels:
        st.info("No graphic novels yet. Click 'New Novel' to get started!")
        render_feedback_link()
        return

    for novel in novels:
        col1, col2 = st.columns([4, 1])
        with col1:
            icon = "✅" if novel['status'] == 'completed' else "⏳" if novel['status'] == 'processing' else "❌"
            if st.button(f"{icon} {novel['title']}", key=f"view_{novel['id']}", type="tertiary"):
                st.query_params["novel_id"] = novel['id']
                navigate_to('novel_detail')
            st.caption(f"{novel['art_style'] or '?'} • {novel.get('page_count', '?')} pages")
        with col2:
            if st.button("🗑️", key=f"del_{novel['id']}"):
                db_delete_novel(novel['id'], st.session_state.user_id)
                st.rerun()
        st.divider()

    render_feedback_link()
