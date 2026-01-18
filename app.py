"""Streamlit frontend for Illustrate AI with user authentication."""
import streamlit as st
import asyncio
from pathlib import Path
from typing import Optional

from constants import ART_STYLES
from utils import calculate_page_count
from config import config, ERA_CONSTRAINTS
from storage.bucket import BucketStorage

# Page Configuration
st.set_page_config(page_title="Illustrative AI", page_icon="📚", layout="centered")

# Hide Streamlit's hamburger menu
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

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
    # Read page from URL query params for initial load
    valid_pages = {'home', 'login', 'register', 'dashboard', 'settings', 'generate', 'novel_detail'}
    url_page = st.query_params.get("page", "home")
    initial_page = url_page if url_page in valid_pages else 'home'

    defaults = {
        # Auth state
        'user_id': None,
        'user_email': None,
        'page': initial_page,  # Read from URL or default to home

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


def db_get_novel(novel_id: str, user_id: str) -> Optional[dict]:
    """Get a single novel by ID."""
    db = get_db()
    if db is None:
        return None
    try:
        novel = db.query(GraphicNovel).filter(
            GraphicNovel.id == novel_id, GraphicNovel.user_id == user_id
        ).first()
        result = novel.to_dict() if novel else None
        db.close()
        return result
    except Exception:
        db.close()
        return None


def get_download_url(storage_key: str) -> Optional[str]:
    """Generate a presigned download URL for a storage key."""
    try:
        storage = BucketStorage()
        return storage.generate_presigned_url(storage_key, expiration=3600)
    except Exception:
        return None


def run_async(coro):
    """Run an async coroutine."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
    render_feedback_link()

    if st.button("← Back to Dashboard", use_container_width=True):
        navigate_to('dashboard')


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


async def resume_pipeline(status, novel: dict, api_key: str):
    """
    Resume a partially completed pipeline from the last checkpoint.
    """
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
            from datetime import datetime
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

                saved_key = db_get_api_key(st.session_state.user_id)

                if saved_key:
                    if st.button("Resume Generation", type="primary", use_container_width=True):
                        config.gemini_api_key = saved_key

                        with st.status("Resuming...", expanded=True) as resume_status:
                            try:
                                db_update_novel(novel_id, status="processing")
                                storage_keys = run_async(resume_pipeline(resume_status, novel, saved_key))

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
                else:
                    st.warning("Add your Gemini API key in Settings to resume this novel.")
                    if st.button("Go to Settings"):
                        navigate_to('settings')

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


def render_generate_page():
    """Generation page."""
    st.title("📚 Illustrative AI")
    st.caption("Transform literature into graphic novels")

    if is_logged_in():
        if st.button("← Dashboard"):
            navigate_to('dashboard')

    st.subheader("1. API Key")
    saved_key = db_get_api_key(st.session_state.user_id) if is_logged_in() else None

    if saved_key:
        st.success("Using saved API key")
        api_key = saved_key
    else:
        api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.api_key)
        st.session_state.api_key = api_key

    st.subheader("2. Upload Manuscript")
    uploaded_file = st.file_uploader("Choose a .txt file (max 10MB)", type=["txt"])

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

        if st.button("🚀 Generate", type="primary", disabled=not api_key or st.session_state.is_running, use_container_width=True):
            st.session_state.is_running = True
            config.gemini_api_key = api_key

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


@st.cache_data
def get_sample_pdf_page(page_num: int) -> bytes:
    """Extract a single page from the sample PDF as PNG bytes."""
    try:
        import pymupdf
    except ImportError:
        return None

    sample_pdf = Path("assets/20-thousand-leagues-under-the-sea.pdf")
    if not sample_pdf.exists():
        return None

    doc = pymupdf.open(str(sample_pdf))
    if page_num >= len(doc):
        doc.close()
        return None

    page = doc[page_num]
    # Render at 2.5x zoom for larger, crisp images
    mat = pymupdf.Matrix(2.5, 2.5)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


@st.cache_data
def get_sample_pdf_page_count() -> int:
    """Get total page count from sample PDF."""
    try:
        import pymupdf
    except ImportError:
        return 0

    sample_pdf = Path("assets/20-thousand-leagues-under-the-sea.pdf")
    if not sample_pdf.exists():
        return 0
    doc = pymupdf.open(str(sample_pdf))
    count = len(doc)
    doc.close()
    return count


def render_home_page():
    """Home page."""
    # Auth buttons at top right
    col_title, col_spacer, col_login, col_register = st.columns([3, 1, 1, 1])
    with col_title:
        st.title("Illustrative AI")
    with col_login:
        if st.button("Login", use_container_width=True):
            navigate_to('login')
    with col_register:
        if st.button("Sign Up", use_container_width=True, type="primary"):
            navigate_to('register')

    st.caption("Transform Literature into Graphic Novels")

    st.markdown("""
An experimental pipeline for generating **100+ page graphic novels** with consistent characters and visual continuity.

Unlike single-image generators, this project tackles the hard problem of **maintaining character appearance,
art style, and narrative coherence** across long-form illustrated content using a coordinated text + image model pipeline.

Upload any public domain book from [Project Gutenberg](https://www.gutenberg.org/) and watch it transform
into a complete graphic novel with consistent character designs, dynamic panel layouts, and proper speech bubbles.

[GitHub Discussions](https://github.com/arvindang/illustrative-public/discussions) · Questions, feedback, and feature requests welcome
    """)

    # Pricing/info section
    st.divider()
    st.markdown("### Free to Try")
    st.info("""
**Create your first graphic novel free** — sign up and start generating.

Full novels (100+ pages) use significant AI compute. Future pricing TBD based on usage.
    """)

    # Sample Preview Section
    st.divider()
    st.markdown("### Sample Output: *20,000 Leagues Under the Sea*")
    st.caption("104 pages · ~$30 USD in API costs · Botanical illustration style")

    page_count = get_sample_pdf_page_count()

    if page_count > 0:
        # Initialize page number in session state
        if 'preview_page' not in st.session_state:
            st.session_state.preview_page = 1

        # Navigation controls
        col_prev, col_slider, col_next = st.columns([1, 6, 1])

        with col_prev:
            if st.button("Prev", use_container_width=True, disabled=st.session_state.preview_page <= 1):
                st.session_state.preview_page -= 1
                st.rerun()

        with col_slider:
            page_num = st.slider(
                "Page",
                min_value=1,
                max_value=page_count,
                value=st.session_state.preview_page,
                key="page_slider",
                label_visibility="collapsed"
            )
            if page_num != st.session_state.preview_page:
                st.session_state.preview_page = page_num

        with col_next:
            if st.button("Next", use_container_width=True, disabled=st.session_state.preview_page >= page_count):
                st.session_state.preview_page += 1
                st.rerun()

        st.caption(f"Page {st.session_state.preview_page} of {page_count}")

        # Display current page
        img_bytes = get_sample_pdf_page(st.session_state.preview_page - 1)
        if img_bytes:
            st.image(img_bytes, use_container_width=True)
        else:
            st.warning("Could not load page preview")
    else:
        st.info("Sample preview not available")

    st.divider()

    # How it works section
    st.markdown("### How It Works")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**")
        st.caption("Upload any public domain book as a .txt file")
    with col2:
        st.markdown("**2. Customize**")
        st.caption("Choose your preferred art style")
    with col3:
        st.markdown("**3. Generate**")
        st.caption("AI creates scripts, illustrations, and layouts")

    st.divider()

    # Technical approach section
    st.markdown("### Technical Approach")
    st.markdown("""
- **3-Agent Pipeline**: Scripting → Illustration → Composition, each stage optimized for its task
- **Character Reference Sheets**: Generates character designs upfront, then passes them as visual references to maintain consistency
- **Context Caching**: Uses Gemini's 2M token context window to keep the full book in memory during script generation
- **Batch-First Architecture**: Designed for 100+ page runs with progress tracking and resume capability
- **Multi-style Support**: Manga, comic book, watercolor, botanical illustration, and more
- **Export**: PDF and EPUB output with proper text overlays (not baked into images)
    """)


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
    elif page == 'settings':
        if not is_logged_in():
            navigate_to('login')
        render_settings_page()
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
