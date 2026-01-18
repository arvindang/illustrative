"""Session state management and navigation utilities."""
import asyncio
import streamlit as st


def init_session_state():
    """Initialize session state variables."""
    # Read page from URL query params for initial load
    valid_pages = {'home', 'login', 'register', 'dashboard', 'generate', 'novel_detail'}
    url_page = st.query_params.get("page", "home")
    initial_page = url_page if url_page in valid_pages else 'home'

    defaults = {
        # Auth state
        'user_id': None,
        'user_email': None,
        'page': initial_page,  # Read from URL or default to home

        # Pipeline state
        'input_path': None,
        'word_count': 0,
        'is_running': False,
        'pipeline_complete': False,
        'output_paths': {},
        'current_novel_id': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_logged_in() -> bool:
    """Check if user is authenticated."""
    return st.session_state.user_id is not None


def navigate_to(page: str):
    """Navigate to a page, updating both session state and URL."""
    st.session_state.page = page
    st.query_params["page"] = page
    st.rerun()


def render_feedback_link():
    """Render a subtle feedback/support link to GitHub Discussions."""
    st.markdown(
        '<p style="text-align: center; color: #888; font-size: 0.85em;">'
        '💬 Questions or feedback? Visit our '
        '<a href="https://github.com/arvindang/illustrative-public/discussions" target="_blank" style="color: #888;">Community Discussions</a>'
        '</p>',
        unsafe_allow_html=True
    )


def run_async(coro):
    """Run an async coroutine."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
