import cv2
import numpy as np
from ultralytics import YOLO
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import matplotlib.pyplot as plt
import os

# ---------- Configuration ----------
IMAGE_PATH = '35.17338462951889_-97.43320995718854_gtd_53.png'
YOLO_MODEL_PATH = 'yolov8n.pt'
SD_MODEL = "stabilityai/stable-diffusion-2-inpainting"

# ---------- Load Models ----------
yolo_model = YOLO(YOLO_MODEL_PATH)
pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL, torch_dtype=torch.float16).to("mps")

# ---------- Load Image ----------
original_img_bgr = cv2.imread(IMAGE_PATH)
original_img = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
height, width = original_img.shape[:2]

# ---------- Run YOLO Detection ----------
results = yolo_model.predict(IMAGE_PATH, imgsz=(height, width), conf=0.25)[0]

# ---------- Filter Vehicle Detections ----------
vehicle_class_ids = [2, 5, 7]  # car, bus, truck
vehicle_detections = []

for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
    if int(cls) in vehicle_class_ids:
        vehicle_detections.append({
            'bbox': box.cpu().numpy(),
            'confidence': float(conf),
            'class_id': int(cls),
            'class_name': yolo_model.names[int(cls)]
        })

print(f"Detected {len(vehicle_detections)} vehicles.")

# ---------- Helper Function: Generate Full-Sized Mask ----------
def generate_full_mask(image_shape, bbox, padding=20):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    xmin, ymin, xmax, ymax = bbox.astype(int)

    # Add padding, ensuring coordinates stay within image boundaries
    xmin = max(xmin - padding, 0)
    ymin = max(ymin - padding, 0)
    xmax = min(xmax + padding, image_shape[1])
    ymax = min(ymax + padding, image_shape[0])

    mask[ymin:ymax, xmin:xmax] = 255
    return mask

# ---------- Directories to Save Outputs ----------
os.makedirs('mask_overlays', exist_ok=True)

# ---------- Inpainting Loop (Full Image + One Mask at a Time) ----------
current_img = original_img.copy()

for idx, detection in enumerate(vehicle_detections):
    print(f"Inpainting {detection['class_name']} (Confidence: {detection['confidence']:.2f})")

    # Generate Full Image Mask for this vehicle
    full_mask = generate_full_mask(current_img.shape[:2], detection['bbox'], padding=20)

    # Save Binary Mask
    mask_save_path = f"mask_overlays/mask_{idx+1}_{detection['class_name']}.png"
    cv2.imwrite(mask_save_path, full_mask)

    # Save Mask Overlay Visualization
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(current_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(current_img)
    plt.imshow(full_mask, cmap='jet', alpha=0.4)
    plt.title("Mask Overlay")
    plt.axis('off')

    overlay_save_path = f"mask_overlays/overlay_{idx+1}_{detection['class_name']}.png"
    plt.savefig(overlay_save_path)
    plt.close()

    print(f"Saved overlay to {overlay_save_path}")

    # Prepare PIL Images for Stable Diffusion (Resize to 512x512)
    img_pil = Image.fromarray(current_img).resize((512, 512), Image.LANCZOS)
    mask_pil = Image.fromarray(full_mask).resize((512, 512), Image.NEAREST)

    # Inpainting with SD
    prompt = "Empty driveway, no vehicles, realistic ground surface, seamless"
    negative_prompt = "cars, vehicles, people, blurry, distorted, text, shadow, watermark, logo"    

    result = pipe(prompt=prompt, negative_prompt=negative_prompt, image=img_pil, mask_image=mask_pil).images[0]

    # Resize Inpainted Result back to Original Image Size
    result_resized = result.resize((width, height), Image.LANCZOS)
    result_np = np.array(result_resized)

    # Update only the masked region in current image
    mask_bool = full_mask.astype(bool)
    current_img[mask_bool] = result_np[mask_bool]

# ---------- Save Final Output ----------
save_path = 'inpainted_fullimage_patchwise.png'
cv2.imwrite(save_path, cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR))
print(f"Final inpainted image saved to {save_path}")