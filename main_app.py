import streamlit as st
import os
import io
import re
import json
import base64
import tempfile
import zipfile
import requests
import numpy as np
from PIL import Image, ImageDraw
from pdf2image import convert_from_bytes

st.set_page_config(page_title="AIマスク")
st.title("AIマスキングアプリ")
st.caption("Created by Kumagif & Co.")

# -----------------------------------------------------
# 🔑 OpenAI REST API クライアント
# -----------------------------------------------------
def ask_ai_for_sensitive_texts(ocr_text):
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY が設定されていません")
        return []

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    prompt = f"""
以下はOCRで抽出されたテキストです。
マイナンバー、基礎年金番号、保険証番号、保険者番号、整理番号など
マスクすべき個人情報の箇所を抽出し、JSONで返してください。

例：
[
  {{"text": "1234-5678-9012", "reason": "マイナンバー"}},
  {{"text": "987654321", "reason": "基礎年金番号"}}
]

OCR結果:
{ocr_text}
"""

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=data)
    content = response.json()["choices"][0]["message"]["content"]

    m = re.search(r"\[.*?\]", content, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            return []
    return []

# -----------------------------------------------------
# 🟦 Google Vision API（REST版）
# -----------------------------------------------------
def get_vision_words(image_bytes):
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ GOOGLE_API_KEY が設定されていません")
        return [], ""

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    payload = {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }
        ]
    }

    r = requests.post(url, json=payload)
    result = r.json()

    if "error" in result:
        st.error("Vision API error: " + result["error"]["message"])
        return [], ""

    words = []
    full_text = result["responses"][0].get("fullTextAnnotation", {}).get("text", "")

    for page in result["responses"][0].get("fullTextAnnotation", {}).get("pages", []):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    text = "".join([s["text"] for s in word.get("symbols", [])]).strip()
                    if not text:
                        continue
                    v = word.get("boundingBox", {}).get("vertices", [])
                    if len(v) >= 4:
                        x1, y1 = v[0].get("x", 0), v[0].get("y", 0)
                        x2, y2 = v[2].get("x", 0), v[2].get("y", 0)
                        words.append({"text": text, "bbox": (x1, y1, x2, y2)})

    return words, full_text

# -----------------------------------------------------
# 🟥 黒塗り処理
# -----------------------------------------------------
def apply_mask(image, words, sensitive_texts):
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)

    margin_x = 20
    margin_y = 12

    # 数字の連続を自動連結
    blocks = []
    tmp = {"text": "", "bbox": None}

    def merge(b1, b2):
        if not b1:
            return b2
        return (
            min(b1[0], b2[0]),
            min(b1[1], b2[1]),
            max(b1[2], b2[2]),
            max(b1[3], b2[3]),
        )

    for w in words:
        if re.match(r"^[0-9０-９\-ー]+$", w["text"]):
            tmp["text"] += w["text"]
            tmp["bbox"] = merge(tmp["bbox"], w["bbox"])
        else:
            if tmp["text"]:
                blocks.append(tmp)
            tmp = {"text": "", "bbox": None}
    if tmp["text"]:
        blocks.append(tmp)

    # マスキング
    for s in sensitive_texts:
        cleaned = re.sub(r"[^0-9]", "", s["text"])
        if not cleaned:
            continue
        for b in blocks:
            b_clean = re.sub(r"[^0-9]", "", b["text"])
            if cleaned in b_clean or cleaned[-4:] in b_clean:
                x1, y1, x2, y2 = b["bbox"]
                draw.rectangle(
                    [
                        x1 - margin_x,
                        y1 - margin_y,
                        x2 + margin_x,
                        y2 + margin_y
                    ],
                    fill="black"
                )
    return im

# -----------------------------------------------------
# PDF → PNG変換
# -----------------------------------------------------
def convert_pdf_to_images(pdf_bytes):
    try:
        images = convert_from_bytes(pdf_bytes)
        out = []
        for i in images:
            buf = io.BytesIO()
            i.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out
    except Exception as e:
        st.error(f"PDF変換エラー: {e}")
        return []

# -----------------------------------------------------
# 画像1枚を処理
# -----------------------------------------------------
def process_image_file(bytes_data):
    words, text = get_vision_words(bytes_data)
    sensitive = ask_ai_for_sensitive_texts(text)

    img = Image.open(io.BytesIO(bytes_data))
    masked = apply_mask(img, words, sensitive)

    buf = io.BytesIO()
    masked.save(buf, format="PNG")
    return buf.getvalue()

# -----------------------------------------------------
# UI
# -----------------------------------------------------
st.subheader("ファイルをアップロードしてください👇")

uploaded_files = st.file_uploader(
    "画像またはPDF（複数可）",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.button("🖤 Vision + AI マスキング実行"):
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "masked.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in uploaded_files:
                st.write(f"処理中: {f.name}")

                if f.name.lower().endswith(".pdf"):
                    pages = convert_pdf_to_images(f.read())
                    for idx, p in enumerate(pages):
                        out = process_image_file(p)
                        zf.writestr(f"{f.name}_page{idx+1}.png", out)
                else:
                    out = process_image_file(f.read())
                    zf.writestr(f"masked_{f.name}", out)

        with open(zip_path, "rb") as fp:
            st.download_button(
                "📦 マスク済ZIPをダウンロード",
                fp.read(),
                file_name="masked_outputs.zip"
            )

    st.success("完了しました！")
