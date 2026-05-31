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
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


interpreter = load_model()

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

    # Sesuaikan input jika model TFLite bertipe quantized
    input_dtype = input_details[0]["dtype"]

    if input_dtype in [np.uint8, np.int8]:
        input_scale, input_zero_point = input_details[0]["quantization"]

        if input_scale > 0:
            img_array = img_array / input_scale + input_zero_point

        img_array = np.clip(
            img_array,
            np.iinfo(input_dtype).min,
            np.iinfo(input_dtype).max
        ).astype(input_dtype)
    else:
        img_array = img_array.astype(np.float32)

    # Prediksi
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])

    # Ambil nilai prediksi mentah
    pred_value = float(prediction.ravel()[0])

    # Jika output TFLite bertipe quantized, kembalikan ke skala asli
    output_dtype = output_details[0]["dtype"]

    if output_dtype in [np.uint8, np.int8, np.int16, np.int32]:
        output_scale, output_zero_point = output_details[0]["quantization"]

        if output_scale > 0:
            pred_value = (pred_value - output_zero_point) * output_scale

    hasil = int(np.round(pred_value))

    if hasil < 0:
        hasil = 0

    st.success(f"Estimasi jumlah benih ikan: {hasil} ekor")
