import streamlit as st
import asyncio
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Import your agents
# from ingestion_agent import IngestionAgent # Removed: File not found and redundant
from scripting_agent import ScriptingAgent
from character_architect import CharacterArchitect
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent
from exporter_agent import ExporterAgent

load_dotenv()

# Page Configuration
st.set_page_config(page_title="LegendLens AI", page_icon="📚", layout="wide")

def init_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'project_config' not in st.session_state:
        st.session_state.project_config = {}
    if 'chapter_map' not in st.session_state:
        st.session_state.chapter_map = None
    if 'beat_sheet_data' not in st.session_state:
        st.session_state.beat_sheet_data = None
    if 'script_data' not in st.session_state:
        st.session_state.script_data = None
    if 'characters_designed' not in st.session_state:
        st.session_state.characters_designed = False
    if 'context_constraints' not in st.session_state:
        st.session_state.context_constraints = "Setting: 1860s. Use period-accurate technology (e.g. Victorian steamships, not modern aircraft carriers). Characters underwater MUST wear steampunk diving suits with copper helmets."

def reset_pipeline():
    st.session_state.step = 1
    st.session_state.beat_sheet_data = None
    st.session_state.script_data = None
    st.session_state.characters_designed = False
    st.rerun()

def main():
    init_session_state()
    
    st.title("📚 LegendLens: Graphic Novel Engine")
    
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Project Settings")
        
        # API Key Check
        if not os.getenv("GEMINI_API_KEY"):
            st.error("❌ GEMINI_API_KEY not found in environment!")
            st.info("Please add it to your .env file.")
            st.stop()
            
        style = st.selectbox("Art Style", 
            ["Lush Watercolor", "Gritty Noir", "Classic Comic Book", "Ukiyo-e Woodblock", "Cyberpunk Neon", "Botanical Illustration"])
        
        tone = st.selectbox("Narrative Tone", 
            ["Heroic", "Suspenseful", "Melancholic", "Whimsical", "Dark Fantasy", "Educational"])
            
        test_mode = st.checkbox("Test Mode (Generate 1 page/segment)", value=True)
        
        st.divider()
        if st.button("🔄 Reset Project"):
            reset_pipeline()
            
    # --- Main Workflow State Machine ---
    
    # STEP 1: UPLOAD & INGESTION
    if st.session_state.step == 1:
        st.header("Step 1: Upload Manuscript")
        st.info("Upload a text file (.txt) to begin the adaptation process.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["txt"])
        
        if uploaded_file:
            # Save file
            input_path = Path("assets/input") / uploaded_file.name
            input_path.parent.mkdir(parents=True, exist_ok=True)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File loaded: {uploaded_file.name}")
            
            if st.button("Generate Beat Sheet ➡️"):
                st.session_state.project_config = {
                    "input_path": str(input_path),
                    "style": style,
                    "tone": tone,
                    "context_constraints": context_constraints,
                    "test_mode": test_mode
                }
                
                # Run Architect Phase
                scripter = ScriptingAgent(str(input_path))
                
                with st.spinner("🗺️ Mapping Chapters & Detecting Context (Gemini 3.0 Flash)..."):
                    full_text = scripter.load_content(test_mode=False)
                    
                    # Parallelize mapping and context analysis
                    chapter_map_task = scripter.generate_chapter_map(full_text)
                    context_task = scripter.analyze_global_context(full_text)
                    
                    chapter_map, detected_context = asyncio.run(asyncio.gather(chapter_map_task, context_task))
                    
                    st.session_state.chapter_map = chapter_map
                    st.session_state.context_constraints = detected_context
                    
                    # Update config with the detected context
                    st.session_state.project_config["context_constraints"] = detected_context

                with st.spinner("🏗️ Architecting Story Arc (Gemini 3.0 Flash)..."):
                    target_pages = 1 if test_mode else 10
                    
                    beat_sheet = asyncio.run(scripter.generate_beat_sheet(
                        full_text,
                        chapter_map=st.session_state.chapter_map, 
                        style=style, 
                        target_page_count=target_pages
                    ))
                    
                    st.session_state.beat_sheet_data = beat_sheet
                    st.session_state.step = 1.5
                    st.rerun()

    # STEP 1.5: BEAT SHEET REVIEW
    elif st.session_state.step == 1.5:
        st.header("Step 1.5: Review Story Arc")
        st.info("Review the page-by-page breakdown before generating the full script.")
        
        beat_sheet = st.session_state.beat_sheet_data
        
        # Simple Table/JSON View
        for beat in beat_sheet:
            with st.expander(f"Page {beat['page_number']}: {beat['narrative_goal']}", expanded=False):
                st.write(f"**Visual:** {beat['key_visual']}")
                st.write(f"**Atmosphere:** {beat['atmosphere']}")
        
        col1, col2 = st.columns([1, 4])
        with col1:
             if st.button("⬅️ Back"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Approve & Write Full Script ➡️"):
                config = st.session_state.project_config
                scripter = ScriptingAgent(config['input_path'])
                full_text = scripter.load_content(test_mode=False)
                chapter_map = st.session_state.chapter_map
                
                with st.spinner("✍️ Writing Scene Scripts in parallel (Gemini 3.0 Flash)..."):
                    # Use the parallelized generate_script or gather write_page_script calls
                    tasks = []
                    for beat in beat_sheet:
                        relevant_chapters = beat.get('relevant_chapters', [])
                        context_text = scripter.get_chapter_text(full_text, chapter_map, relevant_chapters)
                        tasks.append(scripter.write_page_script(
                            beat, context_text, config['style'], config['tone'], config.get('context_constraints', "")
                        ))
                    
                    full_script = asyncio.run(asyncio.gather(*tasks))
                    st.session_state.script_data = full_script
                    
                    # Save locally as expected by next steps
                    suffix = "_test_page.json" if config['test_mode'] else "_full_script.json"
                    output_file = Path("assets/output") / f"{Path(config['input_path']).stem}{suffix}"
                    with open(output_file, "w") as f:
                        json.dump(full_script, f, indent=2)

                    st.session_state.step = 2
                    st.rerun()

    # STEP 2: SCRIPT REVIEW
    elif st.session_state.step == 2:
        st.header("Step 2: Script Review")
        
        # Allow downloading the raw JSON
        script_str = json.dumps(st.session_state.script_data, indent=2)
        st.download_button("Download Script JSON", script_str, "script.json", "application/json")
        
        # Visualizer
        st.subheader("Generated Scenes")
        script_data = st.session_state.script_data
        
        for page in script_data:
            with st.expander(f"Page {page['page_number']} - {page.get('layout_style', 'Standard')}", expanded=True):
                for panel in page['panels']:
                    cols = st.columns([1, 3])
                    with cols[0]:
                        st.markdown(f"**Panel {panel['panel_id']}**")
                        st.caption(f"Chars: {', '.join(panel['characters'])}")
                    with cols[1]:
                        st.text_area("Visual", panel['visual_description'], height=80, key=f"p{page['page_number']}_pan{panel['panel_id']}_vis", disabled=True)
                        st.info(f"🗣️ **Dialogue:** {panel['dialogue']}")
        
        col1, col2 = st.columns([1, 4])
        with col1:
             if st.button("⬅️ Back"):
                st.session_state.step = 1.5 # Go back to Beat Sheet
                st.rerun()
        with col2:
            if st.button("Approve Script & Analyze Characters ➡️"):
                st.session_state.step = 3
                st.rerun()

    # STEP 3: CHARACTER DESIGN
    elif st.session_state.step == 3:
        st.header("Step 3: Character Design")
        
        config = st.session_state.project_config
        # Determine script file path based on logic in ScriptingAgent
        # (Re-deriving path logic here - ideally ScriptingAgent should return the path)
        input_stem = Path(config['input_path']).stem
        suffix = "_test_page.json" if config['test_mode'] else "_full_script.json"
        script_path = Path("assets/output") / f"{input_stem}{suffix}"
        
        architect = CharacterArchitect(str(script_path))
        chars = architect.get_main_characters()
        
        st.write(f"**Found {len(chars)} Main Characters:** {', '.join(chars)}")
        
        if not st.session_state.characters_designed:
            if st.button("🎨 Generate Character Sheets"):
                with st.spinner("Designing characters in parallel..."):
                    asyncio.run(architect.design_all_characters(config['style']))
                st.session_state.characters_designed = True
                st.rerun()
        else:
            # Display Generated Characters
            st.subheader("Character Reference Sheets")
            cols = st.columns(3)
            for i, char in enumerate(chars):
                char_dir = Path("assets/output/characters") / char.lower().replace(" ", "_")
                # Look for ref_0.png
                img_path = char_dir / "ref_0.png"
                with cols[i % 3]:
                    if img_path.exists():
                        st.image(str(img_path), caption=char)
                    else:
                        st.warning(f"No image for {char}")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("⬅️ Back to Script"):
                    st.session_state.step = 2
                    st.rerun()
            with col2:
                if st.button("Approve Characters & Start Production ➡️"):
                    st.session_state.step = 4
                    st.rerun()

    # STEP 4: ILLUSTRATION & COMPOSITION
    elif st.session_state.step == 4:
        st.header("Step 4: Production (Illustration & Composition)")
        
        config = st.session_state.project_config
        input_stem = Path(config['input_path']).stem
        suffix = "_test_page.json" if config['test_mode'] else "_full_script.json"
        script_path = Path("assets/output") / f"{input_stem}{suffix}"
        
        st.info("This process may take some time depending on the number of panels.")
        
        if st.button("🚀 Launch Illustrator Engine"):
            status_container = st.status("🎨 Illustrating Panels...", expanded=True)
            
            # 1. Illustration
            style_prompt = f"{config['style']} style, {config['tone']} tone, high-quality graphic novel art."
            illustrator = IllustratorAgent(str(script_path), style_prompt)
            
            # We wrap the asyncio call
            try:
                asyncio.run(illustrator.run_production())
                status_container.update(label="Illustration Complete!", state="running", expanded=False)
                
                # 2. Composition
                status_container.write("📐 Assembling Final Pages...")
                
                # Check for optimized script (vision-verified bubble placements)
                optimized_script = script_path.parent / f"{script_path.stem}_optimized.json"
                final_script_path = str(optimized_script) if optimized_script.exists() else str(script_path)
                
                compositor = CompositorAgent(final_script_path)
                compositor.run()
                
                status_container.update(label="✅ Production Complete!", state="complete", expanded=False)
                st.session_state.step = 5
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # STEP 5: FINAL PREVIEW & EXPORT
    elif st.session_state.step == 5:
        st.header("🎉 Final Output")
        st.balloons()
        
        final_pages_dir = Path("assets/output/final_pages")
        page_files = sorted(list(final_pages_dir.glob("page_*.png")))
        
        if page_files:
            # Export Options
            st.subheader("📦 Package & Download")
            col_pdf, col_epub = st.columns(2)
            
            config = st.session_state.project_config
            input_stem = Path(config['input_path']).stem
            output_base = Path("assets/output") / input_stem
            
            exporter = ExporterAgent(str(final_pages_dir), str(output_base))
            
            with col_pdf:
                if st.button("Generate PDF 📄"):
                    with st.spinner("Creating PDF..."):
                        pdf_path = exporter.export_pdf()
                        if pdf_path and pdf_path.exists():
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="Download PDF",
                                    data=f,
                                    file_name=pdf_path.name,
                                    mime="application/pdf"
                                )
            
            with col_epub:
                if st.button("Generate EPUB 📚"):
                    with st.spinner("Creating EPUB..."):
                        epub_path = exporter.export_epub(title=input_stem.replace("-", " ").title())
                        if epub_path and epub_path.exists():
                            with open(epub_path, "rb") as f:
                                st.download_button(
                                    label="Download EPUB",
                                    data=f,
                                    file_name=epub_path.name,
                                    mime="application/epub+zip"
                                )

            st.divider()
            st.subheader("Page Preview")
            for page in page_files:
                st.image(str(page), caption=page.name, use_container_width=True)
        else:
            st.warning("No pages found. Something went wrong.")
            
        if st.button("Start New Project"):
            reset_pipeline()

if __name__ == "__main__":
    main()