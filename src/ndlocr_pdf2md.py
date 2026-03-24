"""ndlocr-pdf2md: PDFからMarkdownを直接生成するCLIツール

PDFの各ページを画像として読み込み、ndlocr-liteのOCRエンジンで
文字認識を行い、段落・見出し・注釈を自動検出してMarkdownに変換する。
"""
import sys
sys.setrecursionlimit(5000)
import os
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium

from ndlocr_ocr2md import (
    _ocr_single_page,
    parse_page_xml,
    detect_running_headers,
    detect_chapter_pages,
    detect_note_pages,
    detect_biblio_pages,
    detect_toc_pages,
    infer_metadata,
    convert_to_markdown,
)


def pdf_page_count(pdf_path):
    """PDFのページ数を返す"""
    pdf = pdfium.PdfDocument(pdf_path)
    n = len(pdf)
    pdf.close()
    return n


def pdf_render_page(pdf_path, page_index, dpi=300):
    """PDFの指定ページを1枚だけ画像として返す（メモリ節約）"""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_index]
    bitmap = page.render(scale=dpi / 72)
    pil_image = bitmap.to_pil().convert('RGB')
    pdf.close()
    return pil_image


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

    # 1. ページ数取得
    n_pages = pdf_page_count(args.input)
    print(f"[INFO] PDF: {n_pages} ページ (DPI={args.dpi})")

    # 2. モデル初期化（1回だけ）
    from ocr import get_detector, get_recognizer
    detector = get_detector(args)
    recognizer100 = get_recognizer(args=args)
    recognizer30 = get_recognizer(args=args, weights_path=args.rec_weights30)
    recognizer50 = get_recognizer(args=args, weights_path=args.rec_weights50)

    # 3. 1ページずつ PDF→画像→OCR→中間構造（メモリ節約）
    pages = []
    for i in range(n_pages):
        name = f"page_{i+1:04d}"
        print(f"[{i+1}/{n_pages}] {name}")
        start = time.time()

        pil_image = pdf_render_page(args.input, i, dpi=args.dpi)
        img = np.array(pil_image)
        del pil_image  # メモリ解放

        root, nlines = _ocr_single_page(img, name, detector, recognizer30, recognizer50, recognizer100)
        del img  # メモリ解放

        elapsed = time.time() - start
        print(f"  {nlines} lines, {elapsed:.1f}s")

        pages.append(parse_page_xml(root, i))
        del root  # メモリ解放

    # 4. 自動検出
    running_headers = detect_running_headers(pages)
    chapter_pages = detect_chapter_pages(pages)
    note_indices = detect_note_pages(pages)
    biblio_indices = detect_biblio_pages(pages)
    toc_indices = detect_toc_pages(pages)

    print(f"[INFO] ランニングヘッダ: {len(running_headers)}件検出")
    print(f"[INFO] 章タイトル: {len(chapter_pages)}件検出")
    for idx, title in chapter_pages:
        print(f"  p{idx+1:03d}: {title[:40]}")
    print(f"[INFO] 注釈ページ: {len(note_indices)}件検出")
    print(f"[INFO] 参考文献ページ: {len(biblio_indices)}件検出")
    print(f"[INFO] 目次ページ: {len(toc_indices)}件検出")

    # 5. メタデータ推測（CLI未指定のフィールドを補完）
    inferred = infer_metadata(pages)
    metadata = {
        "title": args.title or inferred.get("title"),
        "author": args.author or inferred.get("author"),
        "year": args.year or inferred.get("year"),
        "publisher": args.publisher or inferred.get("publisher"),
        "isbn": args.isbn or inferred.get("isbn"),
        "type": args.type,
        "tags": args.tags,
    }
    if inferred.get("title") or inferred.get("author"):
        print(f"[INFO] メタデータ推測: title={inferred.get('title', '—')[:30]}, "
              f"author={inferred.get('author', '—')}, "
              f"publisher={inferred.get('publisher', '—')}, "
              f"year={inferred.get('year', '—')}")
    md = convert_to_markdown(
        pages, running_headers, chapter_pages, note_indices,
        biblio_page_indices=biblio_indices,
        toc_page_indices=toc_indices,
        metadata=metadata,
        no_frontmatter=args.no_frontmatter,
    )

    # 6. 出力（既存ファイルは _OLD に退避）
    output_path = Path(args.output)
    if output_path.exists():
        old_path = output_path.with_stem(output_path.stem + "_OLD")
        # _OLD がすでに存在する場合は番号を付けて退避
        if old_path.exists():
            i = 2
            while True:
                old_path = output_path.with_stem(output_path.stem + f"_OLD{i}")
                if not old_path.exists():
                    break
                i += 1
        output_path.rename(old_path)
        print(f"[INFO] 既存ファイルを退避: {old_path.name}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"[INFO] 出力完了: {args.output} ({len(md)} 文字)")


if __name__ == "__main__":
    main()
