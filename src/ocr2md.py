"""ndlocr-md: OCR画像からMarkdownを直接生成するCLIツール"""
import sys
sys.setrecursionlimit(5000)
import os
import re
import glob
import time
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
def _natural_sort_key(s):
    """自然順ソートキー: 'p003.png' < 'p010.png' < 'p0100.png'"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

from deim import DEIM
from parseq import PARSEQ
from yaml import safe_load
from reading_order.xy_cut.eval import eval_xml
from ndl_parser import convert_to_xml_string3
from ocr import RecogLine, get_detector, get_recognizer, process_detector, process_cascade

# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class LineData:
    text: str
    line_type: str  # "本文", "タイトル本文", "キャプション", etc.
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    conf: float = 0.0

@dataclass
class ParagraphData:
    lines: list[LineData] = field(default_factory=list)
    text: str = ""
    is_heading: bool = False

@dataclass
class BlockData:
    block_type: str  # "柱", "ノンブル", "図版", etc.
    text: str = ""

@dataclass
class PageData:
    page_index: int
    image_name: str = ""
    width: int = 0
    height: int = 0
    paragraphs: list[ParagraphData] = field(default_factory=list)
    blocks: list[BlockData] = field(default_factory=list)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

SENTENCE_END = re.compile(r'[。！？!?\)）」』】〉]$')

NOTE_ENTRY_START = re.compile(
    r'^(?:'
    r'[■□図則開聞男脳国川面編犯注]|'
    r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]|'
    r'[=＝]|'
    r'[一二三四五六七八九十百]\s|'
    r'\d{1,3}\s|'
    r'\d{1,3}[a-zA-Zぁ-んァ-ヴ\u4e00-\u9fff]|'
    r'\(\d+\)|'
    r'同前|'
    r'[A-Z][a-z]'
    r')'
)

CHAPTER_PATTERN = re.compile(r'第\s*[\d１-９一二三四五六七八九十]*\s*[章部編]')

IMAGE_EXTENSIONS = {"jpg", "png", "tiff", "jp2", "tif", "jpeg", "bmp"}

# ---------------------------------------------------------------------------
# OCRパイプライン（既存エンジン再利用）
# ---------------------------------------------------------------------------

def _ocr_single_page(img, imgname, detector, recognizer30, recognizer50, recognizer100):
    """1ページ分のOCR処理。numpy配列を受け取り、XML root要素を返す"""
    img_h, img_w = img.shape[:2]

    detections, classeslist = process_detector(
        detector, inputname=imgname, npimage=img,
        outputpath="", issaveimg=False
    )

    resultobj = [dict(), dict()]
    resultobj[0][0] = list()
    for j in range(17):
        resultobj[1][j] = []
    for det in detections:
        xmin, ymin, xmax, ymax = det["box"]
        conf = det["confidence"]
        char_count = det["pred_char_count"]
        if det["class_index"] == 0:
            resultobj[0][0].append([xmin, ymin, xmax, ymax])
        resultobj[1][det["class_index"]].append([xmin, ymin, xmax, ymax, conf, char_count])

    xmlstr = convert_to_xml_string3(img_w, img_h, imgname, classeslist, resultobj)
    xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
    root = ET.fromstring(xmlstr)
    eval_xml(root, logger=None)

    alllineobj = []
    for idx, lineobj in enumerate(root.findall(".//LINE")):
        xmin = int(lineobj.get("X"))
        ymin = int(lineobj.get("Y"))
        line_w = int(lineobj.get("WIDTH"))
        line_h = int(lineobj.get("HEIGHT"))
        try:
            pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT"))
        except:
            pred_char_cnt = 100.0
        lineimg = img[ymin:ymin+line_h, xmin:xmin+line_w, :]
        alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))

    if len(alllineobj) == 0 and len(detections) > 0:
        page = root.find("PAGE")
        for idx, det in enumerate(detections):
            xmin, ymin, xmax, ymax = det["box"]
            line_w = int(xmax - xmin)
            line_h = int(ymax - ymin)
            if line_w > 0 and line_h > 0:
                line_elem = ET.SubElement(page, "LINE")
                line_elem.set("TYPE", "本文")
                line_elem.set("X", str(int(xmin)))
                line_elem.set("Y", str(int(ymin)))
                line_elem.set("WIDTH", str(line_w))
                line_elem.set("HEIGHT", str(line_h))
                line_elem.set("CONF", f"{det['confidence']:0.3f}")
                pred_char_cnt = det.get("pred_char_count", 100.0)
                line_elem.set("PRED_CHAR_CNT", f"{pred_char_cnt:0.3f}")
                lineimg = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
                alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))

    resultlinesall = process_cascade(
        alllineobj, recognizer30, recognizer50, recognizer100, is_cascade=True
    )

    for idx, lineobj in enumerate(root.findall(".//LINE")):
        if idx < len(resultlinesall):
            lineobj.set("STRING", resultlinesall[idx])

    return root, len(resultlinesall)


def run_ocr_pipeline(args, image_paths):
    """画像ファイルリストにOCRを実行し、各ページのXML root要素リストを返す"""
    detector = get_detector(args)
    recognizer100 = get_recognizer(args=args)
    recognizer30 = get_recognizer(args=args, weights_path=args.rec_weights30)
    recognizer50 = get_recognizer(args=args, weights_path=args.rec_weights50)

    results = []
    for i, inputpath in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(inputpath)}")
        start = time.time()
        pil_image = Image.open(inputpath).convert('RGB')
        img = np.array(pil_image)
        imgname = os.path.basename(inputpath)

        root, nlines = _ocr_single_page(img, imgname, detector, recognizer30, recognizer50, recognizer100)

        elapsed = time.time() - start
        print(f"  {nlines} lines, {elapsed:.1f}s")
        results.append(root)

    return results


def run_ocr_pipeline_images(args, pil_images, page_names=None):
    """PIL Imageリストに直接OCRを実行し、各ページのXML root要素リストを返す"""
    detector = get_detector(args)
    recognizer100 = get_recognizer(args=args)
    recognizer30 = get_recognizer(args=args, weights_path=args.rec_weights30)
    recognizer50 = get_recognizer(args=args, weights_path=args.rec_weights50)

    results = []
    for i, pil_image in enumerate(pil_images):
        name = page_names[i] if page_names else f"page_{i+1:04d}"
        print(f"[{i+1}/{len(pil_images)}] {name}")
        start = time.time()
        img = np.array(pil_image.convert('RGB'))

        root, nlines = _ocr_single_page(img, name, detector, recognizer30, recognizer50, recognizer100)

        elapsed = time.time() - start
        print(f"  {nlines} lines, {elapsed:.1f}s")
        results.append(root)

    return results

# ---------------------------------------------------------------------------
# XML→中間構造変換
# ---------------------------------------------------------------------------

def parse_page_xml(root, page_index):
    """XML root要素からPageDataに変換"""
    page_elem = root.find("PAGE")
    if page_elem is None:
        return PageData(page_index=page_index)

    pd = PageData(
        page_index=page_index,
        image_name=page_elem.get("IMAGENAME", ""),
        width=int(page_elem.get("WIDTH", 0)),
        height=int(page_elem.get("HEIGHT", 0)),
    )

    for elem in page_elem:
        tag = elem.tag
        if tag == 'TEXTBLOCK':
            lines = elem.findall(".//LINE")
            if not lines:
                continue
            heading_lines = []
            body_lines = []
            for line in lines:
                s = line.get("STRING", "").strip()
                if not s:
                    continue
                ld = LineData(
                    text=s,
                    line_type=line.get("TYPE", ""),
                    x=int(line.get("X", 0)),
                    y=int(line.get("Y", 0)),
                    width=int(line.get("WIDTH", 0)),
                    height=int(line.get("HEIGHT", 0)),
                    conf=float(line.get("CONF", 0)),
                )
                if ld.line_type == "タイトル本文":
                    heading_lines.append(ld)
                else:
                    body_lines.append(ld)
            if heading_lines:
                text = "".join(l.text for l in heading_lines)
                pd.paragraphs.append(ParagraphData(lines=heading_lines, text=text, is_heading=True))
            if body_lines:
                text = "".join(l.text for l in body_lines)
                pd.paragraphs.append(ParagraphData(lines=body_lines, text=text, is_heading=False))

        elif tag == 'LINE':
            s = elem.get("STRING", "").strip()
            if s:
                ld = LineData(text=s, line_type=elem.get("TYPE", ""))
                is_h = ld.line_type == "タイトル本文"
                pd.paragraphs.append(ParagraphData(lines=[ld], text=s, is_heading=is_h))

        elif tag == 'BLOCK':
            bt = elem.get("TYPE", "")
            pd.blocks.append(BlockData(block_type=bt))

    return pd

# ---------------------------------------------------------------------------
# 自動検出
# ---------------------------------------------------------------------------

def detect_running_headers(pages):
    """3ページ以上で繰り返し出現するテキストを検出"""
    first_texts = Counter()
    for p in pages:
        if p.paragraphs:
            t = p.paragraphs[0].text[:60]
            first_texts[t] += 1
    headers = set()
    for t, cnt in first_texts.items():
        if cnt >= 3:
            headers.add(t)
    return headers


def detect_chapter_pages(pages):
    """章タイトルページを自動検出。(page_index, title)のリストを返す"""
    chapters = []
    for p in pages:
        heading_paras = [para for para in p.paragraphs if para.is_heading]
        body_paras = [para for para in p.paragraphs if not para.is_heading]
        if len(heading_paras) == 0:
            continue

        all_heading_text = " ".join(hp.text for hp in heading_paras)
        heading_line_count = sum(len(hp.lines) for hp in heading_paras)
        body_line_count = sum(len(bp.lines) for bp in body_paras)

        # 条件: 「第N章」「第N部」等のパターンを含み、タイトル本文LINEが複数ある
        if CHAPTER_PATTERN.search(all_heading_text) and heading_line_count >= 2:
            title = "".join(hp.text for hp in heading_paras)
            chapters.append((p.page_index, title))

    return chapters


def detect_note_pages(pages):
    """注釈ページを自動検出"""
    note_indices = set()
    for p in pages:
        if len(p.paragraphs) == 0:
            continue
        # 各段落のテキストが注釈マーカーで始まるかチェック
        marker_count = 0
        total = 0
        for para in p.paragraphs:
            if para.is_heading:
                continue
            total += 1
            # 段落内の最初のLINEで判定
            if para.lines:
                first_line = para.lines[0].text
                if NOTE_ENTRY_START.match(first_line):
                    marker_count += 1
        if total > 0 and marker_count / total >= 0.5:
            note_indices.add(p.page_index)
    return note_indices


def detect_biblio_pages(pages):
    """参考文献ページを自動検出。『書名』パターンの密度で判定"""
    biblio_indices = set()
    for p in pages:
        all_text = "".join(para.text for para in p.paragraphs)
        if not all_text:
            continue
        # 『』の出現回数が多いページは参考文献
        bracket_count = all_text.count('『')
        if bracket_count >= 5 and bracket_count / (len(all_text) / 100) >= 1.0:
            biblio_indices.add(p.page_index)
    return biblio_indices


def is_section_heading(para):
    """セクション見出しかどうか判定"""
    if not para.is_heading:
        return False
    if len(para.text) > 80:
        return False
    return True

# ---------------------------------------------------------------------------
# 注釈・参考文献エントリ分割
# ---------------------------------------------------------------------------

# 参考文献エントリの区切りパターン: 年。の後に新しいエントリが始まる
BIBLIO_SPLIT = re.compile(
    r'([年年])([。\.、,])(?='
    r'[^\s\d）\)]'  # 年。/年、の後に文字が続く（次のエントリの著者名）
    r')'
)


def split_note_entries(paragraphs):
    """注釈ページの段落群を注釈エントリ単位に分割"""
    raw_lines = []
    for para in paragraphs:
        for line in para.lines:
            raw_lines.append(line.text)

    if not raw_lines:
        return []

    entries = []
    current = raw_lines[0]
    for line in raw_lines[1:]:
        if NOTE_ENTRY_START.match(line):
            entries.append(current)
            current = line
        else:
            current += line
    entries.append(current)
    return entries


def split_biblio_entries(paragraphs):
    """参考文献ページの段落群を文献エントリ単位に分割"""
    # まず全LINEを結合
    raw_lines = []
    for para in paragraphs:
        for line in para.lines:
            raw_lines.append(line.text)

    if not raw_lines:
        return []

    full_text = "".join(raw_lines)

    # 「年。」の後で分割（年。は残す）
    parts = BIBLIO_SPLIT.sub(r'\1\2\n', full_text)
    entries = [e.strip() for e in parts.split('\n') if e.strip()]
    return entries

# ---------------------------------------------------------------------------
# Markdown変換
# ---------------------------------------------------------------------------

def convert_to_markdown(pages, running_headers, chapter_pages, note_page_indices,
                        biblio_page_indices=None,
                        metadata=None, no_frontmatter=False):
    """PageDataリストからMarkdown文字列を生成"""
    out = []
    meta = metadata or {}

    # YAML frontmatter（Obsidian互換）
    if not no_frontmatter:
        out.append("---")
        if meta.get("title"):
            out.append(f"title: \"{meta['title']}\"")
        if meta.get("author"):
            out.append(f"author: \"{meta['author']}\"")
        if meta.get("year"):
            out.append(f"year: {meta['year']}")
        if meta.get("publisher"):
            out.append(f"publisher: \"{meta['publisher']}\"")
        if meta.get("isbn"):
            out.append(f"isbn: \"{meta['isbn']}\"")
        source_type = meta.get("type", "book")
        out.append(f"type: {source_type}")
        out.append(f"pages: {len(pages)}")
        out.append(f"date_created: \"{__import__('datetime').date.today().isoformat()}\"")
        if meta.get("tags"):
            out.append("tags:")
            for tag in meta["tags"]:
                out.append(f"  - {tag}")
        out.append("---")
        out.append("")

    chapter_indices = {idx for idx, _ in chapter_pages}
    chapter_map = {idx: t for idx, t in chapter_pages}

    carry = ""  # 前ページからの未完結テキスト
    prev_was_notes = False

    for page in pages:
        pi = page.page_index

        # ランニングヘッダ除去: 先頭段落がヘッダなら削除
        filtered_paras = []
        for para in page.paragraphs:
            if para.text[:60] in running_headers:
                continue
            filtered_paras.append(para)

        # === 注釈ページ ===
        if pi in note_page_indices:
            if carry:
                out.append(carry)
                out.append("")
                carry = ""
            if not prev_was_notes:
                out.append("---")
                out.append("")
                out.append("### 注")
                out.append("")
            entries = split_note_entries(filtered_paras)
            for entry in entries:
                out.append(entry)
                out.append("")
            prev_was_notes = True
            continue

        if prev_was_notes:
            out.append("---")
            out.append("")
            prev_was_notes = False

        # === 参考文献ページ ===
        if biblio_page_indices and pi in biblio_page_indices:
            if carry:
                out.append(carry)
                out.append("")
                carry = ""
            entries = split_biblio_entries(filtered_paras)
            for entry in entries:
                out.append(entry)
                out.append("")
            continue

        # === 章タイトルページ ===
        if pi in chapter_indices:
            if carry:
                out.append(carry)
                out.append("")
                carry = ""
            out.append("")
            out.append("# " + chapter_map[pi])
            out.append("")
            # 章タイトルページの本文（リード文）を出力
            for para in filtered_paras:
                if para.is_heading:
                    continue  # タイトル本文はスキップ（既に#で出力済み）
                if CHAPTER_PATTERN.search(para.text):
                    continue
                carry = para.text  # リード文はcarryに
            continue

        # === 通常ページ ===
        for i, para in enumerate(filtered_paras):
            text = para.text
            is_heading = is_section_heading(para)

            # carryと結合
            if carry:
                if is_heading:
                    out.append(carry)
                    out.append("")
                    carry = ""
                else:
                    text = carry + text
                    carry = ""

            # 最後の段落で文末未完結ならcarry
            if i == len(filtered_paras) - 1 and not is_heading and not SENTENCE_END.search(text):
                carry = text
                continue

            if is_heading:
                out.append("## " + text)
            else:
                out.append(text)
            out.append("")

    # 残りのcarry
    if carry:
        out.append(carry)
        out.append("")

    content = '\n'.join(out)
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    return content

# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    import argparse
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="ndlocr-md: OCR to Markdown converter")
    parser.add_argument("--sourcedir", type=str, help="Path to image directory")
    parser.add_argument("--sourceimg", type=str, help="Path to single image file")
    parser.add_argument("--output", type=str, required=True, help="Output Markdown file path (.md)")
    parser.add_argument("--title", type=str, help="書籍タイトル")
    parser.add_argument("--author", type=str, help="著者名")
    parser.add_argument("--year", type=str, help="出版年")
    parser.add_argument("--publisher", type=str, help="出版社")
    parser.add_argument("--isbn", type=str, help="ISBN")
    parser.add_argument("--tags", type=str, nargs="*", help="タグ（複数指定可）")
    parser.add_argument("--type", type=str, default="book", help="資料種別 (book, article, etc.)")
    parser.add_argument("--no-frontmatter", action="store_true", help="YAML frontmatter を出力しない")

    # OCRエンジン引数（既存と同じ）
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

    # 画像パス収集
    raw_paths = []
    if args.sourcedir:
        raw_paths.extend(glob.glob(os.path.join(args.sourcedir, "*")))
    if args.sourceimg:
        raw_paths.append(args.sourceimg)

    image_paths = [p for p in raw_paths if p.rsplit(".", 1)[-1].lower() in IMAGE_EXTENSIONS]
    image_paths = sorted(image_paths, key=_natural_sort_key)

    if not image_paths:
        print("画像が見つかりません。")
        return

    print(f"[INFO] {len(image_paths)} ページを処理します")

    # 1. OCR実行
    xml_roots = run_ocr_pipeline(args, image_paths)

    # 2. XML→中間構造
    pages = []
    for i, root in enumerate(xml_roots):
        pages.append(parse_page_xml(root, i))

    # 3. 自動検出
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

    # 4. Markdown変換
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

    # 5. 出力
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"[INFO] 出力完了: {args.output} ({len(md)} 文字)")


if __name__ == "__main__":
    main()
