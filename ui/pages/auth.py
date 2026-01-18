"""Login and registration pages."""
import streamlit as st

from config import config
from ui.session import navigate_to
from ui.db_operations import db_login, db_register


def render_login_page():
    """Login page."""
    st.title("🔐 Login")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter email and password")
            elif not config.is_admin_email(email):
                st.error("Access restricted. Contact administrator for access.")
            else:
                success, result = db_login(email, password)
                if success:
                    st.session_state.user_id = result
                    st.session_state.user_email = email
                    navigate_to('dashboard')
                else:
                    st.error(result)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Account", use_container_width=True):
            navigate_to('register')
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            navigate_to('home')


def render_register_page():
    """Registration page."""
    st.title("📝 Create Account")

    # Check if registration is restricted
    if config.get_admin_emails():
        st.warning("🔒 Registration is currently invite-only.")
        st.caption("Contact the administrator if you need access.")
        st.divider()
        if st.button("← Back to Home", use_container_width=True):
            navigate_to('home')
        return

    with st.form("register_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please fill in all fields")
            elif password != confirm:
                st.error("Passwords do not match")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters")
            else:
                success, msg = db_register(email, password)
                if success:
                    st.success("Account created! Please login.")
                    navigate_to('login')
                else:
                    st.error(msg)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home", use_container_width=True):
            navigate_to('home')
    with col2:
        if st.button("Already have an account?", use_container_width=True):
            navigate_to('login')
