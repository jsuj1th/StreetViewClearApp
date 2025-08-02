import streamlit as st
from PIL import Image
from inpainting_pipeline import inpaint_vehicles

st.set_page_config(page_title="Street View Vehicle Remover", layout="wide")

st.title("🚗 Street View Vehicle Inpainting App")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# Padding Slider
padding = st.slider("Mask Padding (pixels)", min_value=0, max_value=100, value=20, step=5)

# Process Button
if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Original Image", use_column_width=True)

    if st.button("Remove Vehicles"):
        with st.spinner("Processing... Please wait."):
            output_image = inpaint_vehicles(input_image, padding=padding)

        st.success("Done!")
        st.image(output_image, caption="Inpainted Image", use_column_width=True)

        # Download Button
        st.download_button("Download Result", data=output_image.tobytes(), file_name="inpainted_result.png", mime="image/png")