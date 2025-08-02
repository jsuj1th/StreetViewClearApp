import cv2
import numpy as np
from ultralytics import YOLO
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

# ---------- Load Models Once ----------
YOLO_MODEL_PATH = 'yolov8n.pt'
SD_MODEL = "stabilityai/stable-diffusion-2-inpainting"

yolo_model = YOLO(YOLO_MODEL_PATH)
pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL, torch_dtype=torch.float16).to("mps")

# ---------- Helper Functions ----------
def generate_full_mask(image_shape, bbox, padding=20):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    xmin, ymin, xmax, ymax = bbox.astype(int)
    xmin = max(xmin - padding, 0)
    ymin = max(ymin - padding, 0)
    xmax = min(xmax + padding, image_shape[1])
    ymax = min(ymax + padding, image_shape[0])
    mask[ymin:ymax, xmin:xmax] = 255
    return mask

# ---------- Main Inpainting Function ----------
def inpaint_vehicles(input_image_pil, padding=20):
    original_img = np.array(input_image_pil)
    height, width = original_img.shape[:2]

    # YOLO Detection
    results = yolo_model.predict(original_img, imgsz=(height, width), conf=0.25)[0]

    # Filter Vehicle Detections
    vehicle_class_ids = [2, 5, 7]  # car, bus, truck
    vehicle_detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if int(cls) in vehicle_class_ids:
            vehicle_detections.append({'bbox': box.cpu().numpy(), 'confidence': float(conf), 'class_id': int(cls), 'class_name': yolo_model.names[int(cls)]})

    print(f"Detected {len(vehicle_detections)} vehicles.")

    # Process Detections
    current_img = original_img.copy()
    for idx, detection in enumerate(vehicle_detections):
        full_mask = generate_full_mask(current_img.shape[:2], detection['bbox'], padding=padding)

        # Prepare for SD Inpainting
        img_pil = Image.fromarray(current_img).resize((512, 512), Image.LANCZOS)
        mask_pil = Image.fromarray(full_mask).resize((512, 512), Image.NEAREST)

        prompt = "Empty driveway, no vehicles, realistic ground surface, seamless"
        negative_prompt = "cars, vehicles, people, blurry, distorted, text, shadow, watermark, logo"

        result = pipe(prompt=prompt, negative_prompt=negative_prompt, image=img_pil, mask_image=mask_pil).images[0]

        # Resize Back & Update
        result_resized = result.resize((width, height), Image.LANCZOS)
        result_np = np.array(result_resized)

        mask_bool = full_mask.astype(bool)
        current_img[mask_bool] = result_np[mask_bool]

    return Image.fromarray(current_img)