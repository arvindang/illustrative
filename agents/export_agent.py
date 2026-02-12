"""
Export Agent: PDF/EPUB export functionality.

Handles:
- PDF export with JPEG compression
- EPUB 3 fixed-layout export
- Cloud storage upload (optional)

This agent handles all output format-specific logic.
"""

import io
import uuid
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image

from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
from ebooklib import epub
from config import config


class ExportAgent:
    """
    Exports composed pages to PDF and EPUB formats.

    This agent is stateless and can process any directory of page images.
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize the ExportAgent.

        Args:
            output_dir: Directory containing final page images (page_1.png, etc.)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("assets/output/final_pages")

    async def run(self, output_path: Path, title: str = "Graphic Novel", author: str = "Illustrate AI") -> Dict[str, Path]:
        """
        Export to both PDF and EPUB formats.

        Args:
            output_path: Base path for output files (without extension)
            title: Title for metadata
            author: Author for metadata

        Returns:
            Dict with 'pdf' and 'epub' keys pointing to output paths
        """
        return {
            'pdf': self.export_pdf(output_path),
            'epub': self.export_epub(output_path, title=title, author=author)
        }

    def get_sorted_images(self) -> List[Path]:
        """Returns a list of image paths sorted by page number."""
        images = list(self.output_dir.glob("page_*.png"))
        images.sort(key=lambda x: int(x.stem.split("_")[1]))
        return images

    def export_pdf(self, output_path: Path) -> Optional[Path]:
        """
        Bundles images into a PDF with JPEG compression for smaller file sizes.

        Args:
            output_path: Base path for output (will add .pdf extension)

        Returns:
            Path to the generated PDF, or None if no images found
        """
        images = self.get_sorted_images()
        if not images:
            print("Warning: No images found to export to PDF.")
            return None

        pdf_path = output_path.with_suffix(".pdf")

        with Image.open(images[0]) as first_img:
            img_w, img_h = first_img.size

        c = pdf_canvas.Canvas(str(pdf_path), pagesize=(img_w, img_h))
        c.setTitle(output_path.stem.replace("-", " ").replace("_", " ").title())
        c.setAuthor("Illustrative AI")

        for img_path in images:
            # Convert to JPEG in memory for better compression
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                buffer = io.BytesIO()
                rgb_img.save(buffer, format="JPEG", quality=85, optimize=True)
                buffer.seek(0)
                c.drawImage(ImageReader(buffer), 0, 0, width=img_w, height=img_h)
            c.showPage()

        c.save()
        print(f"PDF exported to {pdf_path}")
        return pdf_path

    def export_epub(self, output_path: Path, title: str = "Graphic Novel", author: str = "Illustrate AI") -> Optional[Path]:
        """
        Bundles images into a Fixed-Layout EPUB 3 with JPEG compression.

        Args:
            output_path: Base path for output (will add .epub extension)
            title: Title for EPUB metadata
            author: Author for EPUB metadata

        Returns:
            Path to the generated EPUB, or None if no images found
        """
        images = self.get_sorted_images()
        if not images:
            print("Warning: No images found to export to EPUB.")
            return None

        epub_path = output_path.with_suffix(".epub")
        book = epub.EpubBook()

        book.set_identifier(str(uuid.uuid4()))
        book.set_title(title)
        book.set_language('en')
        book.add_author(author)

        # Fixed Layout metadata (EPUB 3)
        book.add_metadata(None, 'meta', 'pre-paginated', {'property': 'rendition:layout'})
        book.add_metadata(None, 'meta', 'portrait', {'property': 'rendition:orientation'})
        book.add_metadata(None, 'meta', 'auto', {'property': 'rendition:spread'})

        spine = ['nav']
        manifest = []

        for i, img_path in enumerate(images):
            page_num = i + 1

            # Convert to JPEG for better compression
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                buffer = io.BytesIO()
                rgb_img.save(buffer, format="JPEG", quality=85, optimize=True)
                img_content = buffer.getvalue()

            epub_img = epub.EpubImage()
            epub_img.file_name = f'images/page_{page_num}.jpg'
            epub_img.media_type = 'image/jpeg'
            epub_img.content = img_content
            book.add_item(epub_img)

            html_content = f"""
            <?xml version="1.0" encoding="UTF-8"?>
            <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
                <head>
                    <title>Page {page_num}</title>
                    <meta name="viewport" content="width={config.page_width}, height={config.page_height}"/>
                    <style>
                        body {{ margin: 0; padding: 0; background-color: #FFFFFF; }}
                        img {{ width: 100%; height: 100%; display: block; }}
                    </style>
                </head>
                <body>
                    <img src="../images/page_{page_num}.jpg" alt="Page {page_num}"/>
                </body>
            </html>
            """

            item = epub.EpubHtml(title=f'Page {page_num}', file_name=f'text/page_{page_num}.xhtml', content=html_content)
            book.add_item(item)
            manifest.append(item)
            spine.append(item)

        book.toc = tuple(manifest)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = spine

        epub.write_epub(str(epub_path), book, {})
        print(f"EPUB exported to {epub_path}")
        return epub_path

    def export_and_upload(self, output_path: Path, novel_id: str, title: str = "Graphic Novel") -> Dict[str, Optional[str]]:
        """
        Export PDF and EPUB, then upload to cloud storage.

        Args:
            output_path: Base path for export files
            novel_id: UUID of the novel record for storage key naming
            title: Title for the EPUB metadata

        Returns:
            Dict with pdf_storage_key and epub_storage_key
        """
        from storage.bucket import get_storage

        # Generate files locally
        pdf_path = self.export_pdf(output_path)
        epub_path = self.export_epub(output_path, title=title)

        result = {"pdf_storage_key": None, "epub_storage_key": None}

        # Upload to bucket if storage is configured
        storage = get_storage()
        if storage.is_configured():
            if pdf_path and pdf_path.exists():
                pdf_key = f"novels/{novel_id}/output.pdf"
                storage.upload_file(str(pdf_path), pdf_key, content_type="application/pdf")
                result["pdf_storage_key"] = pdf_key
                print(f"PDF uploaded to bucket: {pdf_key}")

            if epub_path and epub_path.exists():
                epub_key = f"novels/{novel_id}/output.epub"
                storage.upload_file(str(epub_path), epub_key, content_type="application/epub+zip")
                result["epub_storage_key"] = epub_key
                print(f"EPUB uploaded to bucket: {epub_key}")
        else:
            print("Warning: Bucket not configured - files stored locally only")

        return result


if __name__ == "__main__":
    # Example usage
    agent = ExportAgent(output_dir=Path("assets/output/final_pages"))

    images = agent.get_sorted_images()
    print(f"Found {len(images)} page images")

    if images:
        output_base = Path("assets/output/test_export")
        agent.export_pdf(output_base)
        agent.export_epub(output_base, title="Test Novel")
