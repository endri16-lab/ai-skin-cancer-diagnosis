# src/evaluate_model.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image

# Paths
metadata_path = "data/HAM10000_metadata.csv"
images_dir = "data/images"
model_path = "results/resnet50_skin_model.pth"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Label mapping
df = pd.read_csv(metadata_path)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
lesion_types = sorted(df['dx'].unique())
label_to_index = {label: i for i, label in enumerate(lesion_types)}
index_to_label = {i: label for label, i in label_to_index.items()}
df['label'] = df['dx'].map(label_to_index)

# Validation data
from sklearn.model_selection import train_test_split
_, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class SkinDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

val_data = SkinDataset(val_df, transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Load model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(lesion_types))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Classification report
print("📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=lesion_types))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=lesion_types, yticklabels=lesion_types, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved to /results/")
