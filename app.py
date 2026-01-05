"""Streamlit frontend for Illustrate AI with user authentication."""
import streamlit as st
import asyncio
import httpx
from pathlib import Path
from typing import Optional

from scripting_agent import ScriptingAgent
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent
from constants import ART_STYLES, NARRATIVE_TONES
from utils import calculate_page_count
from config import config

# Page Configuration
st.set_page_config(page_title="Illustrative AI", page_icon="📚", layout="centered")

# API base URL
API_URL = config.api_url


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        # Auth state
        'access_token': None,
        'user': None,
        'page': 'home',  # home, login, register, dashboard, generate

        # Pipeline state
        'api_key': '',
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


def api_request(method: str, endpoint: str, **kwargs) -> Optional[dict]:
    """Make an API request with error handling."""
    try:
        with httpx.Client(timeout=30.0) as client:
            # Add auth header if logged in
            headers = kwargs.pop('headers', {})
            if st.session_state.access_token:
                headers['Authorization'] = f"Bearer {st.session_state.access_token}"

            response = client.request(method, f"{API_URL}{endpoint}", headers=headers, **kwargs)

            if response.status_code == 401:
                # Token expired, logout
                st.session_state.access_token = None
                st.session_state.user = None
                st.session_state.page = 'login'
                return None

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            error = e.response.json().get('detail', 'Bad request')
            st.error(error)
        else:
            st.error(f"API error: {e.response.status_code}")
        return None
    except httpx.RequestError as e:
        st.error(f"Connection error: Unable to reach API server")
        return None


def is_logged_in() -> bool:
    """Check if user is authenticated."""
    return st.session_state.access_token is not None


def run_async(coro):
    """Run an async coroutine."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def execute_pipeline(status, input_path: str, style: str, tone: str, target_pages: int, test_mode: bool, novel_id: str = None):
    """Execute the full pipeline: Scripting -> Illustration -> Composition."""
    input_stem = Path(input_path).stem

    # Step 1: Scripting
    status.write("--- STEP 1/4: SCRIPTING ---")
    status.write("📚 Loading manuscript...")

    scripter = ScriptingAgent(input_path)

    status.write("💾 Caching book content...")
    script = await scripter.generate_script(
        style=f"{style}, {tone}",
        test_mode=test_mode,
        target_page_override=target_pages
    )

    status.write(f"✅ Script complete: {len(script)} pages generated")

    # Determine script path
    suffix = "_test_page.json" if test_mode else "_full_script.json"
    script_path = Path("assets/output") / f"{input_stem}{suffix}"

    # Step 2: Illustration
    status.write("")
    status.write("--- STEP 2/4: ILLUSTRATION ---")

    style_prompt = f"{style} style, {tone} tone, high-quality graphic novel art."
    illustrator = IllustratorAgent(str(script_path), style_prompt)

    status.write("🎨 Generating character & object reference sheets...")
    await illustrator.generate_all_references(style=style)

    status.write("🖼️ Generating panel images...")
    await illustrator.run_production()

    status.write("✅ Illustration complete!")

    # Step 3: Composition
    status.write("")
    status.write("--- STEP 3/4: COMPOSITION ---")
    status.write("📐 Assembling final pages...")

    compositor = CompositorAgent(str(script_path))
    compositor.run()

    status.write("✅ Composition complete!")

    # Step 4: Export
    status.write("")
    status.write("--- STEP 4/4: EXPORT ---")

    output_base = Path("assets/output") / input_stem
    title = input_stem.replace("-", " ").replace("_", " ").title()

    # Use export_and_upload if we have a novel_id (logged in user)
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

    # Store output paths for download
    st.session_state.output_paths = {
        'script_path': str(script_path),
        'output_base': str(output_base),
        'input_stem': input_stem,
        'pdf_path': str(pdf_path) if pdf_path else None,
        'epub_path': str(epub_path) if epub_path else None,
        **storage_keys
    }

    return True


def render_login_page():
    """Render the login page."""
    st.title("🔐 Login")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter email and password")
            else:
                response = api_request(
                    "POST",
                    "/api/auth/login",
                    data={"username": email, "password": password}
                )
                if response:
                    st.session_state.access_token = response['access_token']
                    # Fetch user info
                    user = api_request("GET", "/api/auth/me")
                    if user:
                        st.session_state.user = user
                        st.session_state.page = 'dashboard'
                        st.rerun()

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Account", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()
    with col2:
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state.page = 'generate'
            st.rerun()


def render_register_page():
    """Render the registration page."""
    st.title("📝 Create Account")

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
                response = api_request(
                    "POST",
                    "/api/auth/register",
                    json={"email": email, "password": password}
                )
                if response:
                    st.success("Account created! Please login.")
                    st.session_state.page = 'login'
                    st.rerun()

    st.divider()
    if st.button("← Back to Login", use_container_width=True):
        st.session_state.page = 'login'
        st.rerun()


def render_dashboard():
    """Render the user dashboard with novel list."""
    st.title("📚 My Graphic Novels")

    user = st.session_state.user
    if user:
        st.caption(f"Logged in as {user['email']}")

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("➕ New Novel", use_container_width=True, type="primary"):
            st.session_state.page = 'generate'
            st.rerun()
    with col2:
        if st.button("🔑 API Key Settings", use_container_width=True):
            st.session_state.page = 'settings'
            st.rerun()
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.access_token = None
            st.session_state.user = None
            st.session_state.page = 'home'
            st.rerun()

    st.divider()

    # Fetch novels
    response = api_request("GET", "/api/novels/")
    if response is None:
        return

    novels = response.get('novels', [])

    if not novels:
        st.info("You haven't created any graphic novels yet. Click 'New Novel' to get started!")
        return

    # Display novels as a simple list
    for novel in novels:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                status_icon = "✅" if novel['status'] == 'completed' else "⏳" if novel['status'] == 'processing' else "❌"
                st.write(f"**{status_icon} {novel['title']}**")
                st.caption(f"{novel['art_style'] or 'Unknown style'} • {novel.get('page_count', '?')} pages • {novel['created_at'][:10] if novel.get('created_at') else 'Unknown date'}")

            with col2:
                if novel['has_pdf']:
                    if st.button("📄 PDF", key=f"pdf_{novel['id']}"):
                        url_response = api_request("GET", f"/api/novels/{novel['id']}/download/pdf")
                        if url_response:
                            st.markdown(f"[Download PDF]({url_response['download_url']})")

            with col3:
                if st.button("🗑️", key=f"del_{novel['id']}"):
                    delete_response = api_request("DELETE", f"/api/novels/{novel['id']}")
                    if delete_response:
                        st.success("Deleted!")
                        st.rerun()

            st.divider()


def render_settings_page():
    """Render the API key settings page."""
    st.title("🔑 API Key Settings")

    # Check current API key status
    status_response = api_request("GET", "/api/auth/api-key")

    if status_response and status_response.get('has_api_key'):
        st.success("You have a Gemini API key saved.")
        if st.button("🗑️ Delete Saved API Key"):
            delete_response = api_request("DELETE", "/api/auth/api-key")
            if delete_response:
                st.success("API key deleted")
                st.rerun()
    else:
        st.info("No API key saved. You'll need to enter it each time, or save one below.")

    st.divider()
    st.subheader("Save New API Key")

    with st.form("api_key_form"):
        api_key = st.text_input("Gemini API Key", type="password")
        submitted = st.form_submit_button("Save API Key", use_container_width=True)

        if submitted and api_key:
            response = api_request("PUT", "/api/auth/api-key", json={"api_key": api_key})
            if response:
                st.success("API key saved securely!")
                st.rerun()

    with st.expander("How to get a Gemini API key", expanded=False):
        st.markdown("""
        **Getting your key:**
        1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
        2. Sign in with your Google account
        3. Click **"Create API Key"**
        4. Copy the key and paste it above
        """)

    st.divider()
    if st.button("← Back to Dashboard", use_container_width=True):
        st.session_state.page = 'dashboard'
        st.rerun()


def render_generate_page():
    """Render the generation page."""
    st.title("📚 Illustrative AI: Graphic Novel Engine")
    st.caption("Transform public domain literature into graphic novels using AI")

    # Back button
    if is_logged_in():
        if st.button("← Back to Dashboard"):
            st.session_state.page = 'dashboard'
            st.rerun()

    # --- API Key Input ---
    st.subheader("1. API Key")

    # Check if user has saved API key
    has_saved_key = False
    if is_logged_in():
        status_response = api_request("GET", "/api/auth/api-key")
        has_saved_key = status_response and status_response.get('has_api_key')

    if has_saved_key:
        st.success("Using your saved API key")
        api_key = "SAVED"  # Placeholder - actual key fetched server-side
    else:
        with st.expander("How to get a Gemini API key", expanded=False):
            st.markdown("""
            **Getting your key:**
            1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
            2. Sign in with your Google account
            3. Click **"Create API Key"**
            4. Copy the key and paste it below

            **Pricing:** The Gemini API has a free tier with generous limits.
            """)

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="Paste your API key here..."
        )
        st.session_state.api_key = api_key

        st.caption("🔒 Your API key is **never stored** unless you save it in settings.")

    # --- File Upload ---
    st.subheader("2. Upload Manuscript")
    uploaded_file = st.file_uploader(
        "Choose a text file (.txt)",
        type=["txt"],
        help="Upload a public domain text file to adapt into a graphic novel"
    )

    if uploaded_file:
        # Save file to disk
        input_path = Path("assets/input") / uploaded_file.name
        input_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.input_path = str(input_path)

        # Calculate word count
        content = uploaded_file.getvalue().decode('utf-8')
        word_count = len(content.split())
        st.session_state.word_count = word_count

        st.success(f"Loaded: {uploaded_file.name} ({word_count:,} words)")

        # --- Page Mode Selection ---
        st.subheader("3. Configuration")

        page_mode = st.radio(
            "Page Count",
            ["Quick Preview (10 pages)", "Auto (recommended)"],
            horizontal=True,
            help="Quick Preview is faster for testing. Auto calculates based on book length."
        )

        # Show recommendation for Auto mode
        if page_mode.startswith("Auto"):
            page_calc = calculate_page_count(word_count)
            st.info(f"📄 Recommended: **{page_calc['recommended']} pages** (~{page_calc['estimated_time_minutes']} min)")

        # --- Style & Tone ---
        col1, col2 = st.columns(2)
        with col1:
            style = st.selectbox("Art Style", ART_STYLES, index=0)
        with col2:
            tone = st.selectbox("Narrative Tone", NARRATIVE_TONES, index=0)

        st.divider()

        # --- Generate Button ---
        can_run = bool(api_key) and not st.session_state.is_running

        if not api_key:
            st.warning("Please enter your Gemini API key above to continue.")

        if st.button(
            "🚀 Generate Graphic Novel",
            type="primary",
            disabled=not can_run,
            use_container_width=True
        ):
            st.session_state.is_running = True
            st.session_state.pipeline_complete = False

            # Get actual API key for logged-in users with saved key
            actual_api_key = api_key
            if is_logged_in() and api_key == "SAVED":
                # We need to fetch the key from the user's profile via the API
                # For now, we'll use the session key or require manual entry
                st.error("Server-side API key retrieval not yet implemented. Please enter key manually.")
                st.session_state.is_running = False
                st.stop()

            # Inject API key into config for the agents
            config.gemini_api_key = actual_api_key

            # Create novel record if logged in
            novel_id = None
            title = Path(input_path).stem.replace("-", " ").replace("_", " ").title()

            if is_logged_in():
                novel_response = api_request("POST", "/api/novels/", json={
                    "title": title,
                    "source_filename": uploaded_file.name,
                    "art_style": style,
                    "narrative_tone": tone,
                    "page_count": 10 if page_mode.startswith("Quick") else page_calc['recommended'],
                })
                if novel_response:
                    novel_id = novel_response['id']
                    st.session_state.current_novel_id = novel_id

            # Determine target pages
            if page_mode.startswith("Quick"):
                target_pages = 10
                test_mode = True
            else:
                page_calc = calculate_page_count(word_count)
                target_pages = page_calc['recommended']
                test_mode = False

            # Run pipeline
            with st.status("🚀 Running pipeline...", expanded=True) as status:
                try:
                    run_async(execute_pipeline(
                        status=status,
                        input_path=str(input_path),
                        style=style,
                        tone=tone,
                        target_pages=target_pages,
                        test_mode=test_mode,
                        novel_id=novel_id
                    ))

                    status.update(label="✅ Pipeline Complete!", state="complete")
                    st.session_state.pipeline_complete = True

                    # Update novel status if logged in
                    if novel_id:
                        output_paths = st.session_state.output_paths
                        api_request("PATCH", f"/api/novels/{novel_id}", json={
                            "status": "completed",
                            "pdf_storage_key": output_paths.get('pdf_storage_key'),
                            "epub_storage_key": output_paths.get('epub_storage_key'),
                        })

                except Exception as e:
                    status.update(label=f"❌ Error: {str(e)[:50]}...", state="error")
                    st.error(f"Pipeline failed: {e}")
                    st.info("Check your API key and try again.")

                    # Update novel status to failed
                    if novel_id:
                        api_request("PATCH", f"/api/novels/{novel_id}", json={"status": "failed"})

                finally:
                    st.session_state.is_running = False

    # --- Download Section ---
    if st.session_state.pipeline_complete and st.session_state.output_paths:
        st.divider()
        st.subheader("📦 Download Your Graphic Novel")
        st.success("Your graphic novel is ready!")

        output_info = st.session_state.output_paths
        input_stem = output_info.get('input_stem', 'novel')
        pdf_path = output_info.get('pdf_path')
        epub_path = output_info.get('epub_path')

        col1, col2 = st.columns(2)

        with col1:
            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📄 Download PDF",
                        data=f.read(),
                        file_name=f"{input_stem}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.warning("PDF not available")

        with col2:
            if epub_path and Path(epub_path).exists():
                with open(epub_path, "rb") as f:
                    st.download_button(
                        label="📚 Download EPUB",
                        data=f.read(),
                        file_name=f"{input_stem}.epub",
                        mime="application/epub+zip",
                        use_container_width=True
                    )
            else:
                st.warning("EPUB not available")

        st.divider()
        if st.button("🔄 Start New Project", use_container_width=True):
            # Reset state
            st.session_state.pipeline_complete = False
            st.session_state.output_paths = {}
            st.session_state.input_path = None
            st.session_state.current_novel_id = None
            st.rerun()


def render_home_page():
    """Render the home/landing page."""
    st.title("📚 Illustrative AI")
    st.subheader("Transform Literature into Graphic Novels")

    st.markdown("""
    **Illustrative AI** uses advanced AI to transform public domain literature
    into beautifully illustrated graphic novels.

    ---

    **Features:**
    - Upload any public domain text
    - Choose from multiple art styles
    - AI-generated consistent character designs
    - Export to PDF and EPUB

    ---
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 Login", use_container_width=True, type="primary"):
            st.session_state.page = 'login'
            st.rerun()
    with col2:
        if st.button("🎨 Try as Guest", use_container_width=True):
            st.session_state.page = 'generate'
            st.rerun()


def main():
    init_session_state()

    # Route to appropriate page
    page = st.session_state.page

    if page == 'home':
        render_home_page()
    elif page == 'login':
        render_login_page()
    elif page == 'register':
        render_register_page()
    elif page == 'dashboard':
        if not is_logged_in():
            st.session_state.page = 'login'
            st.rerun()
        render_dashboard()
    elif page == 'settings':
        if not is_logged_in():
            st.session_state.page = 'login'
            st.rerun()
        render_settings_page()
    elif page == 'generate':
        render_generate_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
