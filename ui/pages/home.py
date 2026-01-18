"""Home page with sample preview."""
from pathlib import Path

import streamlit as st

from ui.session import navigate_to


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
