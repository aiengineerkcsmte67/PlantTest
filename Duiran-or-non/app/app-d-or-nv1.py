import streamlit as st
import os
import cv2
import numpy as np
import joblib
from keras.models import Sequential, load_model
from PIL import Image
import tensorflow as tf

# --- ฟังก์ชันการทำงานหลักของระบบ ---

def extract_glcm_features_from_upload(uploaded_image):
    imggg = np.array(uploaded_image)
    img_rgb = cv2.cvtColor(imggg, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    image_batch = np.expand_dims(img_normalized, axis=0)
    return image_batch




st.set_page_config(page_title="ตรวจสอบใบพืชเบื้องต้น", layout="wide")
st.title("🍀ระบบ01 ตรวจสอบใบพืช🍀")
st.write("""
    อัปโหลดภาพใบไม้ เพื่อใช้ปัญญาประดิษฐ์ประมวลผล
    **คำเตือน:** เป็นเพียงการทดสอบเพื่อเป็นส่วนหนึ่งของตรวจพืชทุเรียน
""")

def load_model_for_app():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None 
    )

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "model", "Duria-or-nonv1_weights.weights.h5")
    model.load_weights(model_path)
    return model

try:
    model = load_model_for_app()
except FileNotFoundError:
    st.error("รายละเอียด Error:", e)
    st.stop()


st.header("อัปโหลดภาพของคุณ")
uploaded_file = st.file_uploader("เลือกไฟล์ภาพ🍀...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.header("ภาพที่อัปโหลด")
        st.image(image, caption="ภาพใบพืช🍀", use_column_width=True)

    # สร้างปุ่มเพื่อเรียกใช้ฟังก์ชันการวินิจฉัย
    if st.sidebar.button("ทำการวินิจฉัย"):
        with st.spinner('กำลังประมวลผลและวิเคราะห์ภาพ...'):
            # สกัดฟีเจอร์จากภาพ
            features = extract_glcm_features_from_upload(image)
            prediction = model.predict(features)

            with col2:
                st.header("ผลการวินิจฉัย")
                if prediction[0][0] > 0.5:
                    st.error("ใบไรโช้")
                    st.info(f"ผลการวินิจฉัยมั่นใจ: {prediction[0][0]*100:.2f}%")
                else:
                    st.success("ใบทุเรียน")
                    st.info(f"ผลการวินิจฉัยมั่นใจ: {100 - prediction[0][0]*100:.2f}%")
                    
else:
    st.info("กรุณาอัปโหลดรูปภาพที่แถบด้านข้างเพื่อเริ่มการวินิจฉัย")