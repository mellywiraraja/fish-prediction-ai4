import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

from ai_edge_litert.interpreter import Interpreter


st.set_page_config(
    page_title="Fish Counter AI",
    page_icon="🐟",
    layout="centered"
)

st.title("🐟 Fish Counter AI")
st.write("Upload gambar benih ikan, lalu sistem akan memperkirakan jumlah benih ikan.")


# Link model TFLite dari Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1zpEbb30FK4sugpBCzi_Ijm8tnu2MPbkL"
MODEL_PATH = "fish_counter_model.tflite"


@st.cache_resource
def load_model():
    # Download model jika belum ada
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # Load model TFLite
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    return interpreter


# Load interpreter
interpreter = load_model()

# Ambil detail input-output model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


uploaded_file = st.file_uploader(
    "Upload gambar benih ikan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang di-upload",
        use_container_width=True
    )

    # Preprocessing gambar
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Sesuaikan tipe input dengan model TFLite
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]

        if scale > 0:
            img_array = img_array / scale + zero_point

        img_array = img_array.astype(np.uint8)
    else:
        img_array = img_array.astype(np.float32)

    # Prediksi
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])

    hasil = int(np.round(float(prediction.ravel()[0])))

    if hasil < 0:
        hasil = 0

    st.success(f"Estimasi jumlah benih ikan: {hasil} ekor")
