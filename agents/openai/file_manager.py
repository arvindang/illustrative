"""
OpenAI Files API manager for reference image uploads.

Uploads reference images (character/object sheets) to the OpenAI Files API
so they can be referenced by file_id in batch /v1/images/edits requests.

Maintains a persistent mapping (name -> file_id) at
assets/output/openai_file_mapping.json for resume capability.
"""
import json
from pathlib import Path
from typing import Dict, Optional

from agents.openai.client import get_openai_client


class OpenAIFileManager:
    """
    Manages reference image uploads to OpenAI's Files API.

    Uploaded files can be referenced by file_id in /v1/images/edits
    requests for character consistency. The mapping is persisted to
    disk for resume support across interrupted runs.
    """

    def __init__(self, mapping_path: Path = None):
        """
        Args:
            mapping_path: Path to persist the name->file_id mapping.
                Defaults to assets/output/openai_file_mapping.json.
        """
        self.mapping_path = mapping_path or Path("assets/output/openai_file_mapping.json")
        self._mapping: Dict[str, str] = self._load_mapping()

    def _load_mapping(self) -> Dict[str, str]:
        """Load the persisted file_id mapping from disk."""
        if self.mapping_path.exists():
            with open(self.mapping_path, "r") as f:
                return json.load(f)
        return {}

    def _save_mapping(self):
        """Save the current file_id mapping to disk."""
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_path, "w") as f:
            json.dump(self._mapping, f, indent=2)

    def get_file_id(self, name: str) -> Optional[str]:
        """
        Look up a previously uploaded file's ID by name.

        Args:
            name: The reference name (e.g., character name).

        Returns:
            The OpenAI file_id, or None if not uploaded.
        """
        return self._mapping.get(name.lower().strip())

    async def upload_reference_image(self, name: str, image_path: Path) -> str:
        """
        Upload a single reference image to OpenAI's Files API.

        Args:
            name: Logical name for the reference (e.g., "captain_nemo").
            image_path: Path to the PNG file.

        Returns:
            The OpenAI file_id.
        """
        key = name.lower().strip()

        # Check if already uploaded
        if key in self._mapping:
            print(f"   Reusing uploaded file for '{name}': {self._mapping[key]}")
            return self._mapping[key]

        client = get_openai_client()

        with open(image_path, "rb") as f:
            upload = await client.files.create(
                file=(image_path.name, f),
                purpose="vision",
            )

        self._mapping[key] = upload.id
        self._save_mapping()
        print(f"   Uploaded reference for '{name}': {upload.id}")
        return upload.id

    async def upload_all_references(
        self,
        char_dir: Path,
        obj_dir: Path = None,
    ) -> Dict[str, str]:
        """
        Upload all character and object reference images.

        Scans the directory structure for ref_0.png files in each
        character/object subfolder and uploads them.

        Args:
            char_dir: Path to the characters directory.
            obj_dir: Optional path to the objects directory.

        Returns:
            Dict mapping name -> file_id for all uploaded references.
        """
        uploaded = {}

        # Upload character references
        if char_dir.exists():
            for char_folder in sorted(char_dir.iterdir()):
                if not char_folder.is_dir():
                    continue
                ref_path = char_folder / "ref_0.png"
                if ref_path.exists():
                    file_id = await self.upload_reference_image(
                        char_folder.name, ref_path
                    )
                    uploaded[char_folder.name] = file_id

        # Upload object references
        if obj_dir and obj_dir.exists():
            for obj_folder in sorted(obj_dir.iterdir()):
                if not obj_folder.is_dir():
                    continue
                ref_path = obj_folder / "ref_0.png"
                if ref_path.exists():
                    file_id = await self.upload_reference_image(
                        obj_folder.name, ref_path
                    )
                    uploaded[obj_folder.name] = file_id

        print(f"   Total references uploaded: {len(uploaded)}")
        return uploaded

    async def cleanup_files(self):
        """
        Delete all uploaded files from OpenAI to free storage.
        Call this after batch jobs are complete.
        """
        client = get_openai_client()
        deleted = 0

        for name, file_id in list(self._mapping.items()):
            try:
                await client.files.delete(file_id)
                deleted += 1
            except Exception as e:
                print(f"   Warning: Failed to delete file {file_id} ({name}): {e}")

        self._mapping.clear()
        self._save_mapping()
        print(f"   Cleaned up {deleted} uploaded files")
