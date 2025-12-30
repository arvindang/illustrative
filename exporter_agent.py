import os
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from ebooklib import epub
import uuid

class ExporterAgent:
    def __init__(self, images_dir: str, output_base_path: str):
        """
        :param images_dir: Directory containing the final page PNGs (page_1.png, page_2.png, ...)
        :param output_base_path: Base path/filename for the output (e.g., assets/output/my_novel)
        """
        self.images_dir = Path(images_dir)
        self.output_base_path = Path(output_base_path)
        
    def get_sorted_images(self):
        """Returns a list of image paths sorted by page number."""
        images = list(self.images_dir.glob("page_*.png"))
        # Sort by the number in the filename (e.g., page_1.png -> 1)
        images.sort(key=lambda x: int(x.stem.split("_")[1]))
        return images

    def export_pdf(self):
        """Bundles images into a PDF."""
        images = self.get_sorted_images()
        if not images:
            print("⚠️ No images found to export to PDF.")
            return None
            
        pdf_path = self.output_base_path.with_suffix(".pdf")
        
        # We'll use the size of the first image as the PDF page size
        with Image.open(images[0]) as first_img:
            img_w, img_h = first_img.size
            
        c = canvas.Canvas(str(pdf_path), pagesize=(img_w, img_h))
        
        for img_path in images:
            # ReportLab uses coordinates from bottom-left
            c.drawImage(str(img_path), 0, 0, width=img_w, height=img_h)
            c.showPage()
            
        c.save()
        print(f"✅ PDF exported to {pdf_path}")
        return pdf_path

    def export_epub(self, title="Graphic Novel", author="LegendLens AI"):
        """
        Bundles images into a Fixed-Layout EPUB 3.
        Note: Fixed Layout is complex in EPUB. This implementation provides a basic version.
        """
        images = self.get_sorted_images()
        if not images:
            print("⚠️ No images found to export to EPUB.")
            return None
            
        epub_path = self.output_base_path.with_suffix(".epub")
        book = epub.EpubBook()
        
        # Set metadata
        book.set_identifier(str(uuid.uuid4()))
        book.set_title(title)
        book.set_language('en')
        book.add_author(author)
        
        # Set Fixed Layout metadata (EPUB 3)
        book.add_metadata(None, 'meta', 'pre-paginated', {'property': 'rendition:layout'})
        book.add_metadata(None, 'meta', 'landscape', {'property': 'rendition:orientation'})
        book.add_metadata(None, 'meta', 'auto', {'property': 'rendition:spread'})

        spine = ['nav']
        manifest = []
        
        for i, img_path in enumerate(images):
            page_num = i + 1
            
            # Add Image to EPUB
            with open(img_path, 'rb') as f:
                img_content = f.read()
            
            epub_img = epub.EpubImage()
            epub_img.file_name = f'images/page_{page_num}.png'
            epub_img.content = img_content
            book.add_item(epub_img)
            
            # Create XHTML page for each image
            # We use a simple layout that scales the image to fit the viewport
            html_content = f"""
            <?xml version="1.0" encoding="UTF-8"?>
            <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
                <head>
                    <title>Page {page_num}</title>
                    <meta name="viewport" content="width=2400, height=3200"/>
                    <style>
                        body {{ margin: 0; padding: 0; background-color: #FFFFFF; }}
                        img {{ width: 100%; height: 100%; display: block; }}
                    </style>
                </head>
                <body>
                    <img src="../images/page_{page_num}.png" alt="Page {page_num}"/>
                </body>
            </html>
            """
            
            item = epub.EpubHtml(title=f'Page {page_num}', file_name=f'text/page_{page_num}.xhtml', content=html_content)
            book.add_item(item)
            manifest.append(item)
            spine.append(item)

        # Basic Table of Contents
        book.toc = tuple(manifest)
        
        # Add default NCX and Nav
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Set Spine
        book.spine = spine
        
        # Save EPUB
        epub.write_epub(str(epub_path), book, {})
        print(f"✅ EPUB exported to {epub_path}")
        return epub_path

if __name__ == "__main__":
    # Test run
    exporter = ExporterAgent("assets/output/final_pages", "assets/output/final_graphic_novel")
    exporter.export_pdf()
    exporter.export_epub()
