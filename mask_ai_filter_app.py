import streamlit as st
def add_custom_css():
    st.markdown("""
    <style>
        /* 全体の余白 */
        .main {
            padding: 2rem;
        }
        /* タイトルを中央に */
        h1 {
            text-align: center;
            font-weight: 800;
            margin-bottom: 2rem;
        }
        /* 領域をカード風に */
        .stFileUploader, .stButton, .stDownloadButton {
            background: #ffffffaa; 
            padding: 1.2rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        /* ボタンの強調 */
        .stButton button {
            border-radius: 10px;
            height: 3rem;
            font-size: 1.1rem;
        }
        /* アップロード箱の美化 */
        .uploadedFile {
            border: 2px dashed #aaa !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            background: #fafafa;
        }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

import os
import io
import re
import json
import tempfile
import zipfile
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw
from pdf2image import convert_from_bytes
from google.cloud import vision

# =========================================
# 画面設定
# =========================================
st.set_page_config(page_title="AIマスク", page_icon="🖤")
st.title("AIマスキングアプリ")
st.caption("Created by Kumagif＆Co.")

st.write("### このアプリについて")
st.markdown(
    """
このアプリは **Google Cloud Vision API + OpenAI（GPT）+ 独自ルール** のハイブリッド構成で、  
マイナンバー・基礎年金番号・保険証番号・保険者番号・整理番号などの **個人情報らしき数列を自動検出して黒塗り** します。

**構成イメージ：**

| 機能 | 技術 | 説明 |
|------|------|------|
| OCR強化 | Google Cloud Vision API | 日本語もレイアウトもかなり正確にテキスト＆座標を抽出 |
| AIフィルタ | OpenAI GPT-4o mini（REST） | OCRテキスト全体を読み、「どこが個人情報か」をテキストベースで判断 |
| マスキング | Pillow＋座標計算 | 数字ブロックを結合し、周辺に余白をとって**太めの黒塗り** |
| PDF対応 | pdf2image＋poppler | PDFをページごとに画像化して同じ処理を実施 |
| UI | Streamlit | ブラウザからまとめてアップロード→ZIPダウンロード |

📌 **注意**：完璧なセキュリティ製品ではなく、下書きチェック用ツールの位置づけです。  
最終版は必ず人の目でも確認してください。
"""
)

st.write("----")

# =========================================
# OpenAI（REST）呼び出し設定
# =========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("⚠ OPENAI_API_KEY が環境変数に設定されていません。`export OPENAI_API_KEY=\"...\"` を実行してください。")


def ask_ai_for_sensitive_texts(ocr_text: str):
    """
    OCR全文を渡して、「怪しい個人情報（番号）」候補を JSON で返してもらう。
    返却形式 例:
    [
      {"text": "1234-5678-9012", "reason": "マイナンバー"},
      {"text": "9876543210", "reason": "基礎年金番号"}
    ]
    """
    if not OPENAI_API_KEY:
        return []

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""
以下は、各種申告書・住民票・保険証などを OCR したテキストです。

マイナンバー、基礎年金番号、保険証記号・番号、保険者番号、整理番号、住民票コード など、
「個人を一意に識別する番号と思われる箇所」をすべて抽出してください。

- 出力は必ず JSON 配列のみ。
- 各要素は `text`（実際の文字列）、`reason`（なぜ危険かの日本語説明）の2フィールド。
- 数字以外（記号やスペース）が混じっていても、そのまま text に入れてよいです。

OCR結果:
----------------
{ocr_text}
----------------
    """.strip()

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # JSON 部分だけ抜き出す
        m = re.search(r"\[.*\]", content, re.S)
        if not m:
            return []
        parsed = json.loads(m.group(0))
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception as e:
        st.error(f"❌ OpenAI呼び出しエラー: {e}")
        return []


# =========================================
# Vision API（OCR）
# =========================================
def get_vision_client():
    try:
        client = vision.ImageAnnotatorClient()
        return client
    except Exception as e:
        st.error(f"❌ Vision API クライアント初期化エラー: {e}")
        return None


def get_vision_words(image_bytes: bytes):
    """
    画像バイト列から
    - 単語ごとの text と bbox（x1, y1, x2, y2）
    - 全文テキスト
    を返す
    """
    client_v = get_vision_client()
    if client_v is None:
        return [], ""

    try:
        image = vision.Image(content=image_bytes)
        response = client_v.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(response.error.message)

        words = []
        full_text = response.full_text_annotation.text

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        text = "".join([s.text for s in word.symbols]).strip()
                        if not text:
                            continue
                        v = word.bounding_box.vertices
                        x1, y1 = v[0].x, v[0].y
                        x2, y2 = v[2].x, v[2].y
                        words.append({"text": text, "bbox": (x1, y1, x2, y2)})

        return words, full_text
    except Exception as e:
        st.error(f"❌ Vision API エラー: {e}")
        return [], ""


# =========================================
# マスキング処理
# =========================================
def apply_mask(image: Image.Image, words, sensitive_texts):
    """
    - Visionの word 群から「数字だけのブロック」をまとめる
    - AIが返した suspicious text と数字だけ比較
    - マッチするブロックの周囲を太めに黒塗り
    """
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)

    # 黒塗りの太さ（周辺余白）
    margin_x = 35  # ←少し太めに
    margin_y = 18

    # 数字ブロックを連結
    blocks = []
    tmp = {"text": "", "bbox": None}

    def merge_bbox(b1, b2):
        if not b1:
            return b2
        x1 = min(b1[0], b2[0])
        y1 = min(b1[1], b2[1])
        x2 = max(b1[2], b2[2])
        y2 = max(b1[3], b2[3])
        return (x1, y1, x2, y2)

    for w in words:
        # 数字とハイフンっぽいものだけを連結
        if re.match(r"^[0-9０-９\-ー−]+$", w["text"]):
            tmp["text"] += w["text"]
            tmp["bbox"] = merge_bbox(tmp["bbox"], w["bbox"])
        else:
            if tmp["text"]:
                blocks.append(tmp)
            tmp = {"text": "", "bbox": None}
    if tmp["text"]:
        blocks.append(tmp)

    # マスキング対象の候補数列を準備
    target_numbers = []
    for s in sensitive_texts:
        t = s.get("text", "")
        cleaned = re.sub(r"[^0-9]", "", t)
        if len(cleaned) >= 4:  # 下4桁マスクもしたいので4桁以上は候補に
            target_numbers.append(cleaned)

    # 実際に黒塗り
    for block in blocks:
        b_clean = re.sub(r"[^0-9]", "", block["text"])
        if not b_clean:
            continue

        # どれかの候補と一致 or 下4桁が一致したらマスク
        need_mask = False
        for num in target_numbers:
            if num and (num in b_clean or b_clean.endswith(num[-4:]) or num.endswith(b_clean[-4:])):
                need_mask = True
                break

        if need_mask and block["bbox"]:
            x1, y1, x2, y2 = block["bbox"]
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(im.width, x2 + margin_x)
            y2 = min(im.height, y2 + margin_y)
            draw.rectangle([x1, y1, x2, y2], fill="black")

    return im


# =========================================
# PDF をページごとに PNG に変換
# =========================================
def convert_pdf_to_images(file_bytes: bytes):
    try:
        images = convert_from_bytes(file_bytes)
        out = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out
    except Exception as e:
        st.error(f"❌ PDF変換エラー: {e}")
        return []


# =========================================
# 画像1枚を処理
# =========================================
def process_image_file(file_bytes: bytes):
    words, ocr_text = get_vision_words(file_bytes)
    if not ocr_text:
        # OCRが完全に失敗した場合はそのまま返す
        return file_bytes

    sensitive = ask_ai_for_sensitive_texts(ocr_text)

    img = Image.open(io.BytesIO(file_bytes))
    masked = apply_mask(img, words, sensitive)

    buf = io.BytesIO()
    masked.save(buf, format="PNG")
    return buf.getvalue()


# =========================================
# Streamlit UI 部分
# =========================================
st.subheader("📂 ファイルをアップロード")

uploaded_files = st.file_uploader(
    "画像またはPDFをまとめてアップロードできます（jpg / jpeg / png / pdf）",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
)

st.markdown(
    """
**使い方：**

1. PDF or 画像（スクショでもOK）を複数選択してアップロード  
2. 下のボタンからマスキング実行  
3. ダウンロードされる ZIP の中に、黒塗り済みPNGが入っています  

※ マイナンバーなどが「下4桁だけ残っている問題」を避けるため、  
数字ブロック単位で**やや広めに黒く塗りつぶす**ように調整しています。
"""
)

if uploaded_files and st.button("🖤 マスキング実行"):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "masked_outputs_ai.zip")

        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in uploaded_files:
                st.write(f"処理中: {f.name}")
                raw = f.read()

                if f.name.lower().endswith(".pdf"):
                    pages = convert_pdf_to_images(raw)
                    for idx, p in enumerate(pages):
                        out_bytes = process_image_file(p)
                        zf.writestr(f"masked_{os.path.splitext(f.name)[0]}_p{idx+1}.png", out_bytes)
                else:
                    out_bytes = process_image_file(raw)
                    zf.writestr(f"masked_{f.name}", out_bytes)

        with open(zip_path, "rb") as fp:
            st.download_button(
                "📦 加工済み ZIP をダウンロード",
                data=fp.read(),
                file_name="masked_outputs_ai.zip",
            )

    st.success("✅ すべてのファイルを処理しました！")
else:
    st.info("まずはファイルをアップロードしてください。")
