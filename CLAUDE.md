# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

国立国会図書館が開発した軽量OCRアプリケーション（ndlocr-lite）。GPU不要でCPUのみで動作し、日本語の書籍・雑誌画像からテキストを抽出する。CLI版とGUI版（Flet）がある。

## コマンド

### インストール・実行（CLI）
```bash
pip install -r requirements.txt
cd src
python3 ocr.py --sourcedir <画像ディレクトリ> --output <出力先>
python3 ocr.py --sourceimg <画像ファイル> --output <出力先>
```

### uv経由
```bash
uv tool install .
ndlocr-lite --sourcedir <画像ディレクトリ> --output <出力先>
```

### GUI ビルド（Flutter SDK 3.27.4 + flet==0.27.6 必要）
```bash
cd ndlocr-lite-gui
python3 -m venv ocrenv && source ./ocrenv/bin/activate
pip install flet==0.27.6
cp -r ../src .
flet build macos  # or windows / linux
```

### テスト
専用テストスイートなし。`resource/` 配下のサンプル画像とその出力XML/JSONで動作確認する。

## アーキテクチャ

### OCRパイプライン（src/ocr.py が起点）

```
入力画像 → レイアウト検出(DEIM) → XML構造生成 → 読み順解析(XY-cut) → 文字認識(PARSeq cascade) → 出力(XML/JSON/TXT)
```

### 主要モジュール

| ファイル | 役割 |
|---------|------|
| `src/ocr.py` | CLIエントリポイント。パイプライン全体の制御。`process()` がメイン処理 |
| `src/deim.py` | DEIMv2レイアウト検出モデルのラッパー。17種のオブジェクトを検出 |
| `src/parseq.py` | PARSeq文字認識モデルのラッパー。画像前処理→推論 |
| `src/ndl_parser.py` | XML解析とデータ構造定義。`Category`列挙型で17種のレイアウトカテゴリを管理 |
| `src/tablerecog.py` | 表認識モジュール |
| `src/reading_order/xy_cut/eval.py` | XY-cutアルゴリズムによる読み順決定 |
| `src/reading_order/order/reorder.py` | 行の並び替えロジック |
| `src/config/ndl.yaml` | レイアウト検出クラスマッピング |
| `src/config/NDLmoji.yaml` | 文字セット定義（認識対象文字） |
| `ndlocr-lite-gui/main.py` | GUIエントリポイント（Flet） |

### カスケード認識

文字認識は3つのPARSeqモデルを文字数に応じて使い分ける：
- 30文字モデル（width=256）→ 短いテキスト行
- 50文字モデル（width=384）→ 中程度のテキスト行
- 100文字モデル（width=768）→ 長いテキスト行

98文字を超える行は分割して再処理する（`process_cascade()`）。

### 推論エンジン

全モデルはONNX形式（`src/model/`配下）で、onnxruntimeで推論。`--device cuda` でGPU利用も可能（onnxruntime-gpu必要）。

### 出力形式

各画像に対し以下を生成：
- `.xml` - バウンディングボックス・テキスト・信頼度スコア付きの完全なOCR結果
- `.json` - 構造化されたテキスト領域情報
- `.txt` - 読み順に従ったプレーンテキスト（縦書き50%超の場合は逆順）
- `viz_*` - 検出結果の可視化画像（`--viz True` 時）

### 実装上の注意点

- 画像は正方形にパディングしてから処理（アスペクト比維持）
- `ThreadPoolExecutor` で文字認識を行単位で並列化
- 縦書き検出：縦書き行が50%超の場合、読み順を反転
- XML解析が空結果の場合、検出結果からLINE要素を合成するフォールバックあり

## ライセンス

CC BY 4.0（Creative Commons Attribution 4.0 International）
