# import streamlit as st
# import cv2
# import numpy as np
# import joblib
# import tensorflow as tf


# def extract_features(uploaded_image):
#     img = image.load_img(uploaded_image, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array


# MODEL_FILENAME = 'model/Duria-or-nonv1.keras'
# model = tf.keras.models.load_model(MODEL_FILENAME)

# image = "testmodel/D1.jpg"

# features = extract_features(image)
# prediction_probs = model.predict(features)
# confidence = np.max(prediction_probs) * 100

# print(confidence)

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# --- ฟังก์ชันการทำงานหลักของระบบ ---

def extract_glcm_features_from_upload(uploaded_image):
    imggg = np.array(uploaded_image)
    img_rgb = cv2.cvtColor(imggg, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized / 255.0
    image_batch = np.expand_dims(img_normalized, axis=0)
    return image_batch

def load_model(model_path):
    # เปลี่ยนจาก model = joblib.load(model_path) เป็นด้านล่างนี้
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# --- การสร้างหน้าเว็บแอปพลิเคชัน (Front-end) ---

# ตั้งค่าหัวข้อและคำอธิบายของหน้าเว็บ
st.set_page_config(page_title="ระบบวินิจฉัยความเสี่ยงมะเร็งผิวหนัง", layout="wide")
st.title("🔬 ระบบวินิจฉัยความเสี่ยงโรคมะเร็งผิวหนัง (Melanoma) จากภาพไฝ")
st.write("""
    อัปโหลดภาพถ่ายไฝของคุณ เพื่อให้ปัญญาประดิษฐ์ช่วยประเมินความเสี่ยงเบื้องต้น
    **คำเตือน:** ผลลัพธ์จากระบบนี้เป็นเพียงการประเมินเบื้องต้นเท่านั้น โปรดปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อการวินิจฉัยที่ถูกต้อง
""")

# โหลดโมเดล
MODEL_FILENAME = 'C:\SMTEProJect\plant\Duiran-or-non\model\Duria-or-nonv1.keras'
try:
    model = load_model(MODEL_FILENAME)
except FileNotFoundError:
    st.error(f"ไม่พบไฟล์โมเดล '{MODEL_FILENAME}'! กรุณาตรวจสอบว่าได้รันสคริปต์ train_model.py ก่อน")
    st.stop()

# สร้างส่วนสำหรับอัปโหลดไฟล์ภาพ
st.sidebar.header("อัปโหลดภาพของคุณ")
uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ภาพไฝ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.header("ภาพที่อัปโหลด")
        st.image(image, caption="ภาพไฝ", use_column_width=True)

    # สร้างปุ่มเพื่อเรียกใช้ฟังก์ชันการวินิจฉัย
    if st.sidebar.button("ทำการวินิจฉัย"):
        with st.spinner('กำลังประมวลผลและวิเคราะห์ภาพ...'):
            # สกัดฟีเจอร์จากภาพ
            features = extract_glcm_features_from_upload(image)
            
            # ทำนายผลด้วยโมเดล
            prediction_proba = model.predict_proba(features)[0]
            
            # หาความน่าจะเป็นของคลาส "Melanoma" (สมมติว่าคลาส 1 คือ Melanoma)
            risk_percentage = prediction_proba[1] * 100
            
            # แสดงผลลัพธ์การวินิจฉัย
            with col2:
                st.header("ผลการวินิจฉัย")
                st.metric(label="ความเสี่ยงที่จะเป็น Melanoma", value=f"{risk_percentage:.2f} %")

                if risk_percentage > 50: # สามารถปรับ Threshold ได้ตามความเหมาะสม
                    st.error("ใบไรโช้")
                    
                else:
                    st.success("ใบทุเรียน")
                    
else:
    st.info("กรุณาอัปโหลดรูปภาพที่แถบด้านข้างเพื่อเริ่มการวินิจฉัย")