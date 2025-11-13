# ==============================================
# AIマスキングアプリ（Vision API + GPT-4o-mini）
# PDF対応版・UI付き
# ==============================================

# ==============================================
# AIマスキングアプリ（Vision API REST + GPT-4o-mini）
# ==============================================
import streamlit as st
import os, io, re, json, base64, tempfile, zipfile, requests, numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI



# -----------------------------
# Google Cloud Vision API (REST版)
# -----------------------------
def get_vision_words(image_bytes):
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ GOOGLE_API_KEY が設定されていません。Streamlit Secrets に追加してください。")
        return [], ""

    try:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

        request_body = {
            "requests": [
                {
                    "image": {"content": image_base64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                }
            ]
        }

        response = requests.post(endpoint, json=request_body)
        result = response.json()

        if "error" in result:
            st.error(f"❌ Vision API Error: {result['error'].get('message')}")
            return [], ""

        words = []
        text_annotation = result["responses"][0].get("fullTextAnnotation", {})
        full_text = text_annotation.get("text", "")

        for page in text_annotation.get("pages", []):
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

    except Exception as e:
        st.error(f"❌ Vision API 呼び出しエラー: {e}")
        return [], ""

# ==============================================
# UI設定
# ==============================================
st.set_page_config(page_title="AIマスク")
st.title("AIマスキングアプリ")
st.caption("Created by Kumagif＆Co.")

mask_style = st.radio("マスク方法を選択", ["黒塗り", "モザイク"], horizontal=True)

uploaded_files = st.file_uploader(
    "画像またはPDFファイルをアップロード（複数可）",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)

st.write("まずは画像またはPDFファイルをアップロードしてください。")

st.markdown("""
---
### 💡このアプリについて
このアプリは **AI＋OCRハイブリッド構成** により、極めて高精度に個人情報を検出しマスキングします。
""")

# =============================================
# API クライアント設定
# ==============================================
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    st.error(f"❌ OpenAI クライアント初期化に失敗しました: {e}")
def get_vision_client():
    try:
        return vision.ImageAnnotatorClient()
    except Exception as e:
        st.error(f"❌ Vision API クライアント初期化エラー: {e}")
        raise e

# ==============================================
# OCR（Vision API）
# ==============================================
def get_vision_words(image_bytes):
    try:
        client_v = get_vision_client()
        image = vision.Image(content=image_bytes)
        response = client_v.document_text_detection(image=image)
        if response.error.message:
            raise Exception(response.error.message)
        words = []
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
        return words, response.full_text_annotation.text
    except Exception as e:
        st.error(f"❌ Vision API エラー: {e}")
        return [], ""

# ==============================================
# GPTで個人情報を抽出
# ==============================================
def ask_ai_for_sensitive_texts(ocr_text):
    prompt = f"""
以下はOCRで抽出されたテキストです。
マイナンバー、保険証番号、保険者番号、基礎年金番号、住民票コードなど
「個人情報に該当する可能性がある箇所」をJSON形式で返してください。

例：
[
  {{"text": "マイナンバー: 1234-5678-9012", "reason": "マイナンバー"}}
]

OCR結果:
{ocr_text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content
        json_match = re.search(r"\[.*\]", content, re.S)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            st.warning("⚠️ AI出力がJSON形式ではありません。")
            return []
    except Exception as e:
        st.error(f"❌ OpenAI API呼び出しエラー: {e}")
        return []

# ==============================================
# 黒塗り処理
# ==============================================
def apply_mask(image, words, sensitive_texts, mask_style="黒塗り"):
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)

    margin_x = 25  # ← 幅を少し太めに
    margin_y = 14

    combined_blocks = []
    temp_block = {"text": "", "bbox": None}

    def merge_bbox(b1, b2):
        if not b1:
            return b2
        x1 = min(b1[0], b2[0])
        y1 = min(b1[1], b2[1])
        x2 = max(b1[2], b2[2])
        y2 = max(b1[3], b2[3])
        return (x1, y1, x2, y2)

    for w in words:
        if re.match(r"^[0-9０-９\-ー]+$", w["text"]):
            temp_block["text"] += w["text"]
            temp_block["bbox"] = merge_bbox(temp_block["bbox"], w["bbox"])
        else:
            if temp_block["text"]:
                combined_blocks.append(temp_block)
            temp_block = {"text": "", "bbox": None}
    if temp_block["text"]:
        combined_blocks.append(temp_block)

    for s in sensitive_texts:
        s_clean = re.sub(r"[^0-9]", "", s["text"])
        if not s_clean:
            continue
        for block in combined_blocks:
            b_clean = re.sub(r"[^0-9]", "", block["text"])
            if s_clean in b_clean or s_clean[-8:] in b_clean or s_clean[-4:] in b_clean:
                x1, y1, x2, y2 = block["bbox"]
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(im.width, x2 + margin_x)
                y2 = min(im.height, y2 + margin_y)
                draw.rectangle([x1, y1, x2, y2], fill="black")
    return im

# ==============================================
# ファイル処理（PDF対応）
# ==============================================
def convert_pdf_to_images(file_bytes):
    try:
        images = convert_from_bytes(file_bytes)
        img_bytes_list = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes_list.append(buf.getvalue())
        return img_bytes_list
    except Exception as e:
        st.error(f"❌ PDF変換エラー: {e}")
        return []

def process_image_file(file_bytes, mask_style="黒塗り"):
    words, ocr_text = get_vision_words(file_bytes)
    sensitive_texts = ask_ai_for_sensitive_texts(ocr_text)
    img = Image.open(io.BytesIO(file_bytes))
    masked = apply_mask(img, words, sensitive_texts, mask_style)
    buf = io.BytesIO()
    masked.save(buf, format="PNG")
    return buf.getvalue()

# ==============================================
# メイン実行処理
# ==============================================
if uploaded_files:
    if st.button("🖤 Vision + AIでマスキング実行"):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "masked_outputs.zip")

            with zipfile.ZipFile(zip_path, "w") as zf:
                for f in uploaded_files:
                    st.write(f"処理中: {f.name}")
                    data = f.read()

                    if f.name.lower().endswith(".pdf"):
                        pdf_images = convert_pdf_to_images(data)
                        if not pdf_images:
                            st.warning(f"{f.name} のPDF変換に失敗しました。スキップします。")
                            continue
                        for i, img_bytes in enumerate(pdf_images):
                            out_bytes = process_image_file(img_bytes, mask_style)
                            zf.writestr(f"masked_{os.path.splitext(f.name)[0]}_page{i+1}.png", out_bytes)
                    else:
                        out_bytes = process_image_file(data, mask_style)
                        zf.writestr(f"masked_{f.name}", out_bytes)

            with open(zip_path, "rb") as fp:
                st.download_button(
                    "📦 加工済みZIPをダウンロード",
                    data=fp.read(),
                    file_name="masked_outputs_ai.zip"
                )
        st.success("✅ すべてのファイルを処理しました！")
