# src/visualize_images.py

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Paths
metadata_path = "data/HAM10000_metadata.csv"
images_dir = "data/images"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(metadata_path)

# Unique lesion types
lesion_types = df['dx'].unique()

# Show 4 images from each class
for lesion in lesion_types:
    subset = df[df['dx'] == lesion].sample(n=4, random_state=42)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Lesion Type: {lesion}", fontsize=16)

    for i, (idx, row) in enumerate(subset.iterrows()):
        img_path = os.path.join(images_dir, f"{row['image_id']}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title(f"{row['image_id']}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_images_{lesion}.png")
    plt.close()

print("✅ Sample images saved to /results/")
