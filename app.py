import streamlit as st
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Import your agents
from ingestion_agent import IngestionAgent
from scripting_agent import ScriptingAgent
from character_architect import CharacterArchitect
from illustrator_agent import IllustratorAgent
from compositor_agent import CompositorAgent

load_dotenv()

# Page Configuration
st.set_page_config(page_title="LegendLens AI", page_icon="📚", layout="wide")

def main():
    st.title("📚 LegendLens: Graphic Novel Engine")
    st.sidebar.header("🎨 Generation Settings")
    
    # 1. Sidebar Configuration
    style = st.sidebar.selectbox("Art Style", 
        ["Lush Watercolor", "Gritty Noir", "Classic Comic Book", "Ukiyo-e Woodblock", "Cyberpunk Neon"])
    tone = st.sidebar.selectbox("Narrative Tone", 
        ["Heroic", "Suspenseful", "Melancholic", "Whimsical", "Dark Fantasy"])
    test_mode = st.sidebar.checkbox("Test Mode (Generate 1 page only)", value=True)
    
    # 2. File Upload
    uploaded_file = st.file_uploader("Upload your royalty-free book (.txt)", type=["txt"])
    
    if uploaded_file:
        # Save uploaded file to input folder
        input_path = Path("assets/input") / uploaded_file.name
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Loaded: {uploaded_file.name}")

        # 3. Execution Pipeline
        if st.button("🚀 Start Batch Generation"):
            run_pipeline(input_path, style, tone, test_mode)

def run_pipeline(input_path, style, tone, test_mode):
    # We use st.status (new in late 2024/2025) to show nested progress
    with st.status("🎬 Executing Production Pipeline...", expanded=True) as status:
        
        # --- PHASE 1: INGESTION ---
        st.write("🧹 Cleaning and segmenting text...")
        ingestor = IngestionAgent()
        # Note: We skip fetch_book since we have the file locally
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        clean_text = ingestor.clean_text(raw_text)
        # For simplicity in the app, we work with the cleaned file directly
        status.update(label="Phase 1 Complete: Text Cleaned", state="running")

        # --- PHASE 2: SCRIPTING ---
        st.write("🧠 Writing the screenplay (Gemini 1.5 Pro)...")
        scripter = ScriptingAgent(str(input_path))
        # This creates the JSON script in assets/output
        script_data = asyncio.run(scripter.generate_script(style, tone, test_mode))
        script_file = Path(f"assets/output/{input_path.stem}_test_page.json" if test_mode else f"assets/output/{input_path.stem}_full_script.json")
        status.update(label="Phase 2 Complete: Script Generated", state="running")

        # --- PHASE 3: CHARACTER DESIGN ---
        st.write("🎨 Designing the main cast...")
        architect = CharacterArchitect(str(script_file))
        chars = architect.get_main_characters()
        for char in chars:
            st.write(f"  Creating reference for: {char}")
            asyncio.run(architect.design_character(char, style))
        status.update(label="Phase 3 Complete: Characters Designed", state="running")

        # --- PHASE 4: ILLUSTRATION ---
        st.write("🖌️ Illustrating panels (Gemini 3 Pro Image)...")
        # Be descriptive with the style prompt for the best results
        style_prompt = f"{style} style, {tone} tone, high-quality graphic novel art."
        illustrator = IllustratorAgent(str(script_file), style_prompt)
        asyncio.run(illustrator.run_production())
        status.update(label="Phase 4 Complete: Panels Illustrated", state="running")

        # --- PHASE 5: COMPOSITION ---
        st.write("📐 Assembling final pages...")
        compositor = CompositorAgent(str(script_file))
        compositor.run()
        status.update(label="✅ Graphic Novel Ready!", state="complete")

    # 4. Display Results
    st.divider()
    st.header("🖼️ Final Production Preview")
    
    final_pages_dir = Path("assets/output/final_pages")
    page_files = sorted(list(final_pages_dir.glob("page_*.png")))
    
    if page_files:
        for page in page_files:
            st.image(str(page), caption=page.name, use_container_width=True)
    else:
        st.error("No pages were generated. Check your logs/API keys.")

if __name__ == "__main__":
    main()