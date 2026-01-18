import sys
import unittest
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ScriptingAgent, IllustratorAgent, CompositorAgent

class TestAgentIsolation(unittest.TestCase):
    def test_scripting_agent_paths(self):
        base_dir = Path("assets/output/test_user/test_novel")
        agent = ScriptingAgent("dummy.txt", base_output_dir=base_dir)
        self.assertEqual(agent.output_dir, base_dir)

    def test_illustrator_agent_paths(self):
        base_dir = Path("assets/output/test_user/test_novel")
        agent = IllustratorAgent("dummy_script.json", "style", base_output_dir=base_dir)
        self.assertEqual(agent.char_base_dir, base_dir / "characters")
        self.assertEqual(agent.obj_base_dir, base_dir / "objects")
        self.assertEqual(agent.output_base_dir, base_dir / "pages")

    def test_compositor_agent_paths(self):
        base_dir = Path("assets/output/test_user/test_novel")
        agent = CompositorAgent("dummy_script.json", base_output_dir=base_dir)
        self.assertEqual(agent.panels_dir, base_dir / "pages")
        self.assertEqual(agent.output_dir, base_dir / "final_pages")
        
        # Test default fallback
        agent_default = CompositorAgent("dummy_script.json")
        from config import config
        self.assertEqual(agent_default.panels_dir, config.pages_dir)

if __name__ == '__main__':
    unittest.main()
