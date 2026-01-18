"""Streamlit frontend for Illustrate AI with user authentication."""
import sys

import streamlit as st

from models.base import SessionLocal, engine, Base

# Page Configuration
st.set_page_config(page_title="Illustrative AI", page_icon="📚", layout="centered")

# Hide Streamlit's hamburger menu
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Create tables if they don't exist
if engine is not None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:
        pass  # Tables may already exist

# Startup diagnostics for Railway debugging
_db_configured = SessionLocal is not None
print(f"[Startup] Database configured: {_db_configured}", file=sys.stderr)
if not _db_configured:
    print("[Startup] WARNING: DATABASE_URL not set - auth features will be disabled", file=sys.stderr)

# Import UI components after Streamlit config
from ui.session import init_session_state, is_logged_in, navigate_to
from ui.pages.home import render_home_page
from ui.pages.auth import render_login_page, render_register_page
from ui.pages.dashboard import render_dashboard
from ui.pages.generate import render_generate_page
from ui.pages.novel_detail import render_novel_detail_page


def main():
    init_session_state()
    page = st.session_state.page

    if page == 'home':
        render_home_page()
    elif page == 'login':
        render_login_page()
    elif page == 'register':
        render_register_page()
    elif page == 'dashboard':
        if not is_logged_in():
            navigate_to('login')
        render_dashboard()
    elif page == 'generate':
        render_generate_page()
    elif page == 'novel_detail':
        if not is_logged_in():
            navigate_to('login')
        render_novel_detail_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
