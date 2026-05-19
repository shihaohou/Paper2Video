"""Rasterize selected vector PDF figures in place into single-image PDFs.

Tectonic (xdvipdfmx) drops parts of complex TikZ content when embedding
into beamer slides (gradients, hatching, transparent scatter clouds),
producing half-blank figures. Re-saving the figure as a single
rasterized page side-steps the problem while keeping the .pdf filename
so slides.tex needs no edit.

Usage (run from the project root, in an env where PyMuPDF is installed):

    python scripts/flatten_figures.py \
        assets/mypaper/latex_proj/fig/motivation1.pdf \
        assets/mypaper/latex_proj/fig/motivation1_1.pdf \
        assets/mypaper/latex_proj/fig/method2.pdf \
        assets/mypaper/latex_proj/fig/method2-cr.pdf
"""
import argparse
from pathlib import Path

import fitz
from PIL import Image


def flatten(pdf_path: Path, dpi: int = 400) -> None:
    src = fitz.open(str(pdf_path))
    page = src[0]
    rect = page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    src.close()

    # Render to a PIL image, then let PIL write a minimal image-only PDF.
    # PyMuPDF's own .save() produces PDF 1.7 streams that xdvipdfmx can't
    # decompress ("tectonic_flate_decompress() failed"). PIL's PDF backend
    # writes simpler PDFs that tectonic handles without complaint.
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.save(str(pdf_path), "PDF", resolution=float(dpi))
    print(f"flattened {pdf_path.name} ({rect.width:.0f}x{rect.height:.0f} pt @ {dpi} dpi)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="PDF figures to flatten in place")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    for f in args.files:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(path)
        flatten(path, dpi=args.dpi)


if __name__ == "__main__":
    main()
