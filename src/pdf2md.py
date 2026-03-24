"""ndlocr-pdf2md: PDFからMarkdownを直接生成するCLIツール

PDFの各ページを画像として読み込み、ndlocr-liteのOCRエンジンで
文字認識を行い、段落・見出し・注釈を自動検出してMarkdownに変換する。
"""
import sys
sys.setrecursionlimit(5000)
import os
import time
import argparse
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium

from ocr2md import (
    run_ocr_pipeline_images,
    parse_page_xml,
    detect_running_headers,
    detect_chapter_pages,
    detect_note_pages,
    detect_biblio_pages,
    convert_to_markdown,
)


def pdf_to_images(pdf_path, dpi=300):
    """PDFの各ページをPIL Imageに変換する"""
    pdf = pdfium.PdfDocument(pdf_path)
    images = []
    n_pages = len(pdf)
    print(f"[INFO] PDF読み込み: {n_pages} ページ (DPI={dpi})")
    for i in range(n_pages):
        page = pdf[i]
        bitmap = page.render(scale=dpi / 72)
        pil_image = bitmap.to_pil()
        images.append(pil_image)
    pdf.close()
    return images


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="ndlocr-pdf2md: PDF to Markdown converter")
    parser.add_argument("--input", type=str, required=True, help="入力PDFファイルパス")
    parser.add_argument("--output", type=str, required=True, help="出力Markdownファイルパス (.md)")
    parser.add_argument("--dpi", type=int, default=300, help="PDF→画像変換のDPI (default: 300)")

    # 書誌情報
    parser.add_argument("--title", type=str, help="書籍タイトル")
    parser.add_argument("--author", type=str, help="著者名")
    parser.add_argument("--year", type=str, help="出版年")
    parser.add_argument("--publisher", type=str, help="出版社")
    parser.add_argument("--isbn", type=str, help="ISBN")
    parser.add_argument("--tags", type=str, nargs="*", help="タグ（複数指定可）")
    parser.add_argument("--type", type=str, default="book", help="資料種別 (book, article, etc.)")
    parser.add_argument("--no-frontmatter", action="store_true", help="YAML frontmatter を出力しない")

    # OCRエンジン引数
    parser.add_argument("--det-weights", type=str, default=str(base_dir / "model" / "deim-s-1024x1024.onnx"))
    parser.add_argument("--det-classes", type=str, default=str(base_dir / "config" / "ndl.yaml"))
    parser.add_argument("--det-score-threshold", type=float, default=0.2)
    parser.add_argument("--det-conf-threshold", type=float, default=0.25)
    parser.add_argument("--det-iou-threshold", type=float, default=0.2)
    parser.add_argument("--rec-weights30", type=str, default=str(base_dir / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"))
    parser.add_argument("--rec-weights50", type=str, default=str(base_dir / "model" / "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx"))
    parser.add_argument("--rec-weights", type=str, default=str(base_dir / "model" / "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx"))
    parser.add_argument("--rec-classes", type=str, default=str(base_dir / "config" / "NDLmoji.yaml"))
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"PDFが見つかりません: {args.input}")
        return

    # 1. PDF→画像変換
    pil_images = pdf_to_images(args.input, dpi=args.dpi)
    page_names = [f"page_{i+1:04d}" for i in range(len(pil_images))]

    print(f"[INFO] {len(pil_images)} ページを処理します")

    # 2. OCR実行
    xml_roots = run_ocr_pipeline_images(args, pil_images, page_names)

    # 3. XML→中間構造
    pages = [parse_page_xml(root, i) for i, root in enumerate(xml_roots)]

    # 4. 自動検出
    running_headers = detect_running_headers(pages)
    chapter_pages = detect_chapter_pages(pages)
    note_indices = detect_note_pages(pages)
    biblio_indices = detect_biblio_pages(pages)

    print(f"[INFO] ランニングヘッダ: {len(running_headers)}件検出")
    print(f"[INFO] 章タイトル: {len(chapter_pages)}件検出")
    for idx, title in chapter_pages:
        print(f"  p{idx+1:03d}: {title[:40]}")
    print(f"[INFO] 注釈ページ: {len(note_indices)}件検出")
    print(f"[INFO] 参考文献ページ: {len(biblio_indices)}件検出")

    # 5. Markdown変換
    metadata = {
        "title": args.title,
        "author": args.author,
        "year": args.year,
        "publisher": args.publisher,
        "isbn": args.isbn,
        "type": args.type,
        "tags": args.tags,
    }
    md = convert_to_markdown(
        pages, running_headers, chapter_pages, note_indices,
        biblio_page_indices=biblio_indices,
        metadata=metadata,
        no_frontmatter=args.no_frontmatter,
    )

    # 6. 出力
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"[INFO] 出力完了: {args.output} ({len(md)} 文字)")


if __name__ == "__main__":
    main()
