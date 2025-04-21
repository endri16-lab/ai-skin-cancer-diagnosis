# src/train_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Paths
metadata_path = "data/HAM10000_metadata.csv"
images_dir = "data/images"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(metadata_path)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))

# Encode labels (string -> int)
lesion_types = sorted(df['dx'].unique())
label_to_index = {label: idx for idx, label in enumerate(lesion_types)}
df['label'] = df['dx'].map(label_to_index)

# Split into train/val
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
])

# Custom dataset class
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

# Create datasets & loaders
train_data = SkinDataset(train_df, transform=transform)
val_data = SkinDataset(val_df, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(lesion_types))  # Replace final layer
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop (simplified)
num_epochs = 3  # Start small for now
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"📘 Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.3f}, Train Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), os.path.join(results_dir, "resnet50_skin_model.pth"))
print("✅ Model training complete and saved to /results/")
