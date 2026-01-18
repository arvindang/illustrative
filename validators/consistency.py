"""
Consistency Auditing module: Cross-panel character consistency checking.

Includes:
- ConsistencyAuditor: Audits character consistency across panels within a page.
"""

from typing import Dict, List
from dataclasses import dataclass, field
from PIL import Image

from google.genai import types
from config import config
from utils import get_client


# ==============================================================================
# CONSISTENCY AUDITOR - Cross-panel character consistency
# ==============================================================================

@dataclass
class ConsistencyAuditResult:
    """Result of auditing consistency across multiple panels."""
    consistent: bool
    character_issues: Dict[str, List[str]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ConsistencyAuditor:
    """
    Audits character consistency across panels within a page.
    Compares character appearances to detect drift.
    """

    async def audit_page_consistency(
        self,
        page_images: List[Image.Image],
        characters: List[str],
        panel_data: List[dict] = None
    ) -> ConsistencyAuditResult:
        """
        Audit character consistency across panels in a page.

        Args:
            page_images: List of PIL Images (panels from same page)
            characters: List of character names appearing in the page
            panel_data: Optional panel script data for context

        Returns:
            ConsistencyAuditResult with consistency status and issues
        """
        if len(page_images) < 2:
            return ConsistencyAuditResult(consistent=True)

        char_list = ", ".join(characters) if characters else "characters in the panels"

        audit_prompt = f"""
You are a quality assurance expert for graphic novel character consistency.

TASK: Compare character appearances across these panels from the SAME PAGE.

CHARACTERS TO CHECK: {char_list}

For each character appearing in multiple panels, evaluate:
1. FACE CONSISTENCY: Is the face shape, features, and expression style consistent?
2. CLOTHING CONSISTENCY: Is the clothing/costume the same across panels?
3. DISTINCTIVE ITEMS: Are signature items (glasses, weapons, accessories) present?
4. ART STYLE: Is the character drawn in the same art style across panels?
5. COLOR PALETTE: Are character colors consistent?

Respond with a JSON object:
{{
    "consistent": <true if characters are consistent, false otherwise>,
    "character_issues": {{
        "<character_name>": ["<issue 1>", "<issue 2>"]
    }},
    "recommendations": ["<suggestion for improving consistency>"]
}}
"""

        input_contents = [audit_prompt]
        for i, img in enumerate(page_images):
            input_contents.append(f"\n--- PANEL {i + 1} ---")
            input_contents.append(img)

        try:
            response = await get_client().aio.models.generate_content(
                model=config.scripting_model_page_script,
                contents=input_contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "consistent": {"type": "BOOLEAN"},
                            "inconsistent_characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "issue_descriptions": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "recommendations": {"type": "ARRAY", "items": {"type": "STRING"}}
                        },
                        "required": ["consistent"]
                    }
                )
            )

            result = response.parsed
            if result:
                return ConsistencyAuditResult(
                    consistent=result.get('consistent', True),
                    character_issues=result.get('character_issues', {}),
                    recommendations=result.get('recommendations', [])
                )

        except Exception as e:
            print(f"   Consistency audit error: {e}")

        return ConsistencyAuditResult(consistent=True)
