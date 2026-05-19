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


def flatten(pdf_path: Path, dpi: int = 400) -> None:
    src = fitz.open(str(pdf_path))
    page = src[0]
    rect = page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    src.close()

    dst = fitz.open()
    new_page = dst.new_page(width=rect.width, height=rect.height)
    new_page.insert_image(rect, pixmap=pix)
    dst.save(str(pdf_path), deflate=True)
    dst.close()
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
