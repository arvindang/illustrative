"""Streamlit frontend for Illustrate AI with user authentication."""
import streamlit as st
import asyncio
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

# Database imports (direct access, no separate API needed)
from sqlalchemy.orm import Session
from models.base import SessionLocal, engine, Base
from models.user import User
from models.novel import GraphicNovel
from api.dependencies import hash_password, verify_password, create_access_token

# Create tables if they don't exist
if engine is not None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:
        pass  # Tables may already exist

# Startup diagnostics for Railway debugging
import sys
_db_configured = SessionLocal is not None
print(f"[Startup] Database configured: {_db_configured}", file=sys.stderr)
if not _db_configured:
    print("[Startup] WARNING: DATABASE_URL not set - auth features will be disabled", file=sys.stderr)


def get_db() -> Optional[Session]:
    """Get database session if configured."""
    if SessionLocal is None:
        return None
    return SessionLocal()


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        # Auth state
        'user_id': None,
        'user_email': None,
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


def is_logged_in() -> bool:
    """Check if user is authenticated."""
    return st.session_state.user_id is not None


def db_register(email: str, password: str) -> tuple[bool, str]:
    """Register a new user."""
    db = get_db()
    if db is None:
        return False, "Database not configured"

    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            db.close()
            return False, "Email already registered"

        user = User(email=email, password_hash=hash_password(password))
        db.add(user)
        db.commit()
        db.close()
        return True, "Account created"
    except Exception as e:
        db.close()
        return False, str(e)


def db_login(email: str, password: str) -> tuple[bool, str]:
    """Login user."""
    db = get_db()
    if db is None:
        return False, "Database not configured"

    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.password_hash):
            db.close()
            return False, "Invalid email or password"

        user_id = str(user.id)
        db.close()
        return True, user_id
    except Exception as e:
        db.close()
        return False, str(e)


def db_has_api_key(user_id: str) -> bool:
    """Check if user has saved API key."""
    db = get_db()
    if db is None:
        return False
    try:
        user = db.query(User).filter(User.id == user_id).first()
        result = user.has_api_key() if user else False
        db.close()
        return result
    except Exception:
        db.close()
        return False


def db_get_api_key(user_id: str) -> Optional[str]:
    """Get decrypted API key."""
    db = get_db()
    if db is None:
        return None
    try:
        user = db.query(User).filter(User.id == user_id).first()
        key = user.get_gemini_api_key() if user and user.has_api_key() else None
        db.close()
        return key
    except Exception:
        db.close()
        return None


def db_save_api_key(user_id: str, api_key: str) -> bool:
    """Save encrypted API key."""
    db = get_db()
    if db is None:
        return False
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.set_gemini_api_key(api_key)
            db.commit()
        db.close()
        return True
    except Exception:
        db.close()
        return False


def db_delete_api_key(user_id: str) -> bool:
    """Delete API key."""
    db = get_db()
    if db is None:
        return False
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.gemini_api_key_encrypted = None
            db.commit()
        db.close()
        return True
    except Exception:
        db.close()
        return False


def db_list_novels(user_id: str) -> list[dict]:
    """List user's novels."""
    db = get_db()
    if db is None:
        return []
    try:
        novels = db.query(GraphicNovel).filter(
            GraphicNovel.user_id == user_id
        ).order_by(GraphicNovel.created_at.desc()).all()
        result = [n.to_dict() for n in novels]
        db.close()
        return result
    except Exception:
        db.close()
        return []


def db_create_novel(user_id: str, title: str, source_filename: str, art_style: str, tone: str, page_count: int) -> Optional[str]:
    """Create novel record."""
    db = get_db()
    if db is None:
        return None
    try:
        novel = GraphicNovel(
            user_id=user_id, title=title, source_filename=source_filename,
            art_style=art_style, narrative_tone=tone, page_count=page_count, status="processing"
        )
        db.add(novel)
        db.commit()
        db.refresh(novel)
        novel_id = str(novel.id)
        db.close()
        return novel_id
    except Exception:
        db.close()
        return None


def db_update_novel(novel_id: str, **kwargs) -> bool:
    """Update novel."""
    db = get_db()
    if db is None:
        return False
    try:
        novel = db.query(GraphicNovel).filter(GraphicNovel.id == novel_id).first()
        if novel:
            for k, v in kwargs.items():
                if hasattr(novel, k) and v is not None:
                    setattr(novel, k, v)
            db.commit()
        db.close()
        return True
    except Exception:
        db.close()
        return False


def db_delete_novel(novel_id: str, user_id: str) -> bool:
    """Delete novel."""
    db = get_db()
    if db is None:
        return False
    try:
        novel = db.query(GraphicNovel).filter(
            GraphicNovel.id == novel_id, GraphicNovel.user_id == user_id
        ).first()
        if novel:
            db.delete(novel)
            db.commit()
        db.close()
        return True
    except Exception:
        db.close()
        return False


def run_async(coro):
    """Run an async coroutine."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def execute_pipeline(status, input_path: str, style: str, tone: str, target_pages: int, test_mode: bool, novel_id: str = None):
    """Execute the full pipeline."""
    input_stem = Path(input_path).stem

    status.write("--- STEP 1/4: SCRIPTING ---")
    status.write("📚 Loading manuscript...")
    scripter = ScriptingAgent(input_path)

    status.write("💾 Caching book content...")
    script = await scripter.generate_script(
        style=f"{style}, {tone}", test_mode=test_mode, target_page_override=target_pages
    )
    status.write(f"✅ Script complete: {len(script)} pages generated")

    suffix = "_test_page.json" if test_mode else "_full_script.json"
    script_path = Path("assets/output") / f"{input_stem}{suffix}"

    status.write("")
    status.write("--- STEP 2/4: ILLUSTRATION ---")
    style_prompt = f"{style} style, {tone} tone, high-quality graphic novel art."
    illustrator = IllustratorAgent(str(script_path), style_prompt)

    status.write("🎨 Generating character & object reference sheets...")
    await illustrator.generate_all_references(style=style)

    status.write("🖼️ Generating panel images...")
    await illustrator.run_production()
    status.write("✅ Illustration complete!")

    status.write("")
    status.write("--- STEP 3/4: COMPOSITION ---")
    status.write("📐 Assembling final pages...")
    compositor = CompositorAgent(str(script_path))
    compositor.run()
    status.write("✅ Composition complete!")

    status.write("")
    status.write("--- STEP 4/4: EXPORT ---")
    output_base = Path("assets/output") / input_stem
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
            else:
                success, result = db_login(email, password)
                if success:
                    st.session_state.user_id = result
                    st.session_state.user_email = email
                    st.session_state.page = 'dashboard'
                    st.rerun()
                else:
                    st.error(result)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Account", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()


def render_register_page():
    """Registration page."""
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
                success, msg = db_register(email, password)
                if success:
                    st.success("Account created! Please login.")
                    st.session_state.page = 'login'
                    st.rerun()
                else:
                    st.error(msg)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    with col2:
        if st.button("Already have an account?", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()


def render_dashboard():
    """User dashboard."""
    st.title("📚 My Graphic Novels")

    if st.session_state.user_email:
        st.caption(f"Logged in as {st.session_state.user_email}")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("➕ New Novel", use_container_width=True, type="primary"):
            st.session_state.page = 'generate'
            st.rerun()
    with col2:
        if st.button("🔑 API Key", use_container_width=True):
            st.session_state.page = 'settings'
            st.rerun()
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.session_state.page = 'home'
            st.rerun()

    st.divider()

    novels = db_list_novels(st.session_state.user_id)
    if not novels:
        st.info("No graphic novels yet. Click 'New Novel' to get started!")
        return

    for novel in novels:
        col1, col2 = st.columns([4, 1])
        with col1:
            icon = "✅" if novel['status'] == 'completed' else "⏳" if novel['status'] == 'processing' else "❌"
            st.write(f"**{icon} {novel['title']}**")
            st.caption(f"{novel['art_style'] or '?'} • {novel.get('page_count', '?')} pages")
        with col2:
            if st.button("🗑️", key=f"del_{novel['id']}"):
                db_delete_novel(novel['id'], st.session_state.user_id)
                st.rerun()
        st.divider()


def render_settings_page():
    """API key settings."""
    st.title("🔑 API Key Settings")

    has_key = db_has_api_key(st.session_state.user_id)

    if has_key:
        st.success("You have a Gemini API key saved.")
        if st.button("🗑️ Delete Saved API Key"):
            db_delete_api_key(st.session_state.user_id)
            st.rerun()
    else:
        st.info("No API key saved.")

    st.divider()
    with st.form("api_key_form"):
        api_key = st.text_input("Gemini API Key", type="password")
        if st.form_submit_button("Save API Key", use_container_width=True):
            if api_key:
                db_save_api_key(st.session_state.user_id, api_key)
                st.success("Saved!")
                st.rerun()

    st.divider()
    if st.button("← Back to Dashboard", use_container_width=True):
        st.session_state.page = 'dashboard'
        st.rerun()


def render_generate_page():
    """Generation page."""
    st.title("📚 Illustrative AI")
    st.caption("Transform literature into graphic novels")

    if is_logged_in():
        if st.button("← Dashboard"):
            st.session_state.page = 'dashboard'
            st.rerun()

    st.subheader("1. API Key")
    saved_key = db_get_api_key(st.session_state.user_id) if is_logged_in() else None

    if saved_key:
        st.success("Using saved API key")
        api_key = saved_key
    else:
        api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.api_key)
        st.session_state.api_key = api_key

    st.subheader("2. Upload Manuscript")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

    if uploaded_file:
        input_path = Path("assets/input") / uploaded_file.name
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        content = uploaded_file.getvalue().decode('utf-8')
        word_count = len(content.split())
        st.success(f"Loaded: {uploaded_file.name} ({word_count:,} words)")

        st.subheader("3. Configuration")
        page_mode = st.radio("Pages", ["Quick Preview (10)", "Auto"], horizontal=True)

        if page_mode.startswith("Auto"):
            page_calc = calculate_page_count(word_count)
            st.info(f"Recommended: {page_calc['recommended']} pages")

        col1, col2 = st.columns(2)
        with col1:
            style = st.selectbox("Art Style", ART_STYLES)
        with col2:
            tone = st.selectbox("Tone", NARRATIVE_TONES)

        st.divider()

        if st.button("🚀 Generate", type="primary", disabled=not api_key or st.session_state.is_running, use_container_width=True):
            st.session_state.is_running = True
            config.gemini_api_key = api_key

            target_pages = 10 if page_mode.startswith("Quick") else page_calc['recommended']
            test_mode = page_mode.startswith("Quick")
            title = Path(input_path).stem.replace("-", " ").replace("_", " ").title()

            novel_id = None
            if is_logged_in():
                novel_id = db_create_novel(st.session_state.user_id, title, uploaded_file.name, style, tone, target_pages)

            with st.status("🚀 Running...", expanded=True) as status:
                try:
                    run_async(execute_pipeline(status, str(input_path), style, tone, target_pages, test_mode, novel_id))
                    status.update(label="✅ Complete!", state="complete")
                    st.session_state.pipeline_complete = True
                    if novel_id:
                        db_update_novel(novel_id, status="completed",
                                       pdf_storage_key=st.session_state.output_paths.get('pdf_storage_key'),
                                       epub_storage_key=st.session_state.output_paths.get('epub_storage_key'))
                except Exception as e:
                    status.update(label=f"❌ Error", state="error")
                    st.error(str(e))
                    if novel_id:
                        db_update_novel(novel_id, status="failed")
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


def render_home_page():
    """Home page."""
    st.title("📚 Illustrative AI")
    st.subheader("Transform Literature into Graphic Novels")

    st.markdown("""
Transform royalty-free books from [Project Gutenberg](https://www.gutenberg.org/) and other public domain sources
into beautifully illustrated graphic novels using **Google Gemini 3** and **Nano Banana**.
    """)

    st.divider()

    # How it works section
    st.markdown("### How It Works")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**")
        st.caption("Upload any public domain book as a .txt file")
    with col2:
        st.markdown("**2. Customize**")
        st.caption("Choose your art style and narrative tone")
    with col3:
        st.markdown("**3. Generate**")
        st.caption("AI creates scripts, illustrations, and layouts")

    st.divider()

    # Features section
    st.markdown("### Features")
    st.markdown("""
- **Multi-style artwork**: Choose from manga, comic book, watercolor, and more
- **Consistent characters**: AI maintains character appearance across all panels
- **Automated layout**: Dynamic page compositions with speech bubbles and captions
- **Export options**: Download as PDF or EPUB
    """)

    st.divider()

    # API Key info
    st.markdown("### Bring Your Own API Key")
    st.info("""
This app requires a **Google Gemini API key** to generate graphic novels.

Get your free API key at [Google AI Studio](https://aistudio.google.com/apikey).
Your key is encrypted and stored securely when you create an account.
    """)

    st.divider()

    # Auth buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 Login", use_container_width=True, type="primary"):
            st.session_state.page = 'login'
            st.rerun()
    with col2:
        if st.button("📝 Create Account", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()


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
