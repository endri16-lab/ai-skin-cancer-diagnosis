import torch
from torchvision import models, transforms
import torch.nn as nn
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Paths
metadata_path = "data/HAM10000_metadata.csv"
images_dir = "data/images"
model_path = "results/resnet50_skin_model.pth"
output_dir = "results/gradcam"
os.makedirs(output_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(metadata_path)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))

# Label mapping
lesion_types = sorted(df['dx'].unique())
label_to_index = {label: i for i, label in enumerate(lesion_types)}
index_to_label = {i: label for label, i in label_to_index.items()}

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(lesion_types))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Grad-CAM setup
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Pick 1 random image per lesion type
for lesion in lesion_types:
    sample = df[df['dx'] == lesion].sample(1, random_state=42).iloc[0]
    image_path = sample['image_path']
    image_id = sample['image_id']

    # Load and transform image
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Prepare original image for overlay
    img_np = np.array(pil_img.resize((224, 224))) / 255.0  # Normalize to [0, 1]
    img_np = np.float32(img_np)

    # Generate Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Save image
    output_file = os.path.join(output_dir, f"gradcam_{lesion}_{image_id}.png")
    cv2.imwrite(output_file, cam_image[:, :, ::-1])  # RGB to BGR for OpenCV
    print(f"✅ Saved: {output_file}")
