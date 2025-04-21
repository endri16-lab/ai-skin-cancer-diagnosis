# src/eda_metadata.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up paths
metadata_path = "data\HAM10000_metadata.csv"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  # Create 'results/' folder if it doesn't exist

# Load the CSV into a DataFrame
df = pd.read_csv(metadata_path)

# Show basic info about the dataset
print("\n🔍 Basic Info:")
print(df.info())
print("\n🧾 First 5 rows:\n", df.head())

# -------------------------------------------
# 📊 1. Plot diagnosis label counts
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='dx', order=df['dx'].value_counts().index)
plt.title("Diagnosis Distribution")
plt.xlabel("Lesion Type (dx)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/diagnosis_distribution.png")
plt.close()

# -------------------------------------------
# 📊 2. Plot patient sex distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='sex')
plt.title("Patient Sex Distribution")
plt.savefig(f"{output_dir}/sex_distribution.png")
plt.close()

# -------------------------------------------
# 📊 3. Plot age distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['age'].dropna(), kde=True, bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.savefig(f"{output_dir}/age_distribution.png")
plt.close()

# -------------------------------------------
# 📊 4. Plot lesion location distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='localization', order=df['localization'].value_counts().index)
plt.title("Lesion Location Distribution")
plt.savefig(f"{output_dir}/location_distribution.png")
plt.close()

print("\n✅ EDA plots saved to /results/")
