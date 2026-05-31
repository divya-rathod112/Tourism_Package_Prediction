import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi

# Initialize Hugging Face API client using your environment secret
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# Define repository structures (Replace with your actual HF username)
HF_USERNAME = "divyarathod112"
DATASET_REPO = f"{HF_USERNAME}/Tourism-Package-Prediction"

# Read raw dataset directly from the Hugging Face Hub Dataset layer
DATASET_PATH = f"hf://datasets/{DATASET_REPO}/tourism.csv"
print(f"Reading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# Drop index/administrative identifiers if present
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Define explicit nominal/categorical lists to sanitize before model pipeline feeding
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']

for col in categorical_cols:
    if col in df.columns:
        # Fill empty missing entries with a standard string designation
        df[col] = df[col].fillna('Unknown')
        # Use LabelEncoder to preserve feature values cleanly for tracking
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Separate target array split
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

# Execute a stratified split to keep target ratio uniform across training/testing
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data partitioning split complete. Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")

# Convert dataframe partitions to local disk CSV targets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Re-upload split features up onto your Hugging Face Data Repository Hub
artifacts = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_name in artifacts:
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )

print("Stratified features completely synced back to Hugging Face Hub dataset!")
