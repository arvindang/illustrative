import streamlit as st
import asyncio
from pathlib import Path

from scripting_agent import ScriptingAgent
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent
from constants import ART_STYLES, NARRATIVE_TONES
from utils import calculate_page_count
from config import config

# Page Configuration
st.set_page_config(page_title="Illustrative AI", page_icon="📚", layout="centered")


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'api_key': '',
        'input_path': None,
        'word_count': 0,
        'is_running': False,
        'pipeline_complete': False,
        'output_paths': {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def run_async(coro):
    """Run an async coroutine."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def execute_pipeline(status, input_path: str, style: str, tone: str, target_pages: int, test_mode: bool):
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

    status.write("📄 Generating PDF...")
    pdf_path = compositor.export_pdf(output_base)

    status.write("📚 Generating EPUB...")
    epub_path = compositor.export_epub(output_base, title=title)

    status.write("✅ Export complete!")

    # Store output paths for download
    st.session_state.output_paths = {
        'script_path': str(script_path),
        'output_base': str(output_base),
        'input_stem': input_stem,
        'pdf_path': str(pdf_path) if pdf_path else None,
        'epub_path': str(epub_path) if epub_path else None
    }

    return True


def main():
    init_session_state()

    st.title("📚 Illustrative AI: Graphic Novel Engine")
    st.caption("Transform public domain literature into graphic novels using AI")

    # --- API Key Input ---
    st.subheader("1. API Key")

    with st.expander("How to get a Gemini API key", expanded=False):
        st.markdown("""
        **Getting your key:**
        1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
        2. Sign in with your Google account
        3. Click **"Create API Key"**
        4. Copy the key and paste it below

        **Pricing:** The Gemini API has a free tier with generous limits.
        For heavy usage, you may need to enable billing in Google Cloud Console.
        """)

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="Paste your API key here..."
    )
    st.session_state.api_key = api_key

    st.caption("🔒 Your API key is **never stored**. It's held only in your browser session, sent directly to Google's API, and discarded when you close this tab.")

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

            # Inject API key into config for the agents
            config.gemini_api_key = api_key

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
                        test_mode=test_mode
                    ))

                    status.update(label="✅ Pipeline Complete!", state="complete")
                    st.session_state.pipeline_complete = True

                except Exception as e:
                    status.update(label=f"❌ Error: {str(e)[:50]}...", state="error")
                    st.error(f"Pipeline failed: {e}")
                    st.info("Check your API key and try again.")

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
            st.rerun()


if __name__ == "__main__":
    main()
