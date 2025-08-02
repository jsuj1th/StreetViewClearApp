import gradio as gr
from detect_vehicle import inpaint_vehicles

def process_image(image, padding):
    return inpaint_vehicles(image, padding=int(padding))

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(0, 100, value=20, step=5, label="Mask Padding (pixels)")
    ],
    outputs=gr.Image(type="pil", label="Inpainted Image"),
    title="Vehicle Removal Inpainting",
    description="Upload an image to remove vehicles using YOLO + Stable Diffusion Inpainting."
)

iface.launch()