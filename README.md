# 🚗 StreetViewClearApp — Vehicle Removal using YOLO + Stable Diffusion

StreetViewClearApp is a web-based tool to automatically detect and remove vehicles (cars, buses, trucks) from street-view images. It leverages YOLOv8 for object detection and Stable Diffusion Inpainting for generating realistic background replacements where vehicles are removed seamlessly.

---

## 📊 Use Cases
- Clean Google Street View images.
- Remove clutter from urban datasets.
- Visualize clean infrastructure for city planning.
- Prepare clean datasets for generative models (GANs, Diffusion).

---

## 🖥️ How It Works?
1. You upload a street-view image (JPG/PNG).
2. YOLOv8 detects vehicles (Car, Bus, Truck) in the image.
3. For each detected vehicle:
   - A padded mask is generated.
   - Stable Diffusion Inpainting fills that masked area with realistic background textures.
4. The masked region is replaced patch-wise into the original image.
5. You can view the cleaned image and download it.
6. Mask overlays are also saved for visualization/debugging purposes.

---

## 📂 Project Structure
StreetViewClearApp/
├── app.py                    # Streamlit Web App Entry Point
├── inpainting_pipeline.py    # YOLO detection + SD inpainting logic
├── models/                   # (Optional) Custom YOLOv8 weights
├── mask_overlays/            # Generated mask overlays per vehicle
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation


---

## ✨ Features
- Clean Streamlit web UI for file uploads and image preview.
- Detects and removes Cars, Buses, and Trucks.
- Dynamically generates padded masks around detections.
- Saves visualization overlays for each masked region.
- Utilizes Apple's MPS (Metal Performance Shaders) for GPU acceleration.
- Download button for final cleaned images.

---

## 🛠️ Setup Instructions

### Step 1: Clone Repository
```bash
git clone <your_repo_url>
cd StreetViewClearApp
```

### Step 2: Install Python Dependencies
```
pip install -r requirements.txt
```

### Step 3: Launch the Streamlit App
```bash
streamlit run app.py
```
