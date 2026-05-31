import os
import pandas as pd
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Retrieve the secret token from the execution environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is missing. Please set it in your environment or repository secrets.")

# Initialize Hugging Face API client
api = HfApi(token=HF_TOKEN)

# Specify your target dataset repository identity
# TODO: Replace 'YOUR_HF_USERNAME' with your actual Hugging Face profile username
HF_USERNAME = "divyarathod112"
DATASET_REPO = f"{HF_USERNAME}/Tourism-Package-Prediction"
REPO_TYPE = "dataset"

# 1. Ensure the remote data repository exists on Hugging Face Hub
try:
    api.repo_info(repo_id=DATASET_REPO, repo_type=REPO_TYPE)
    print(f"Dataset repository '{DATASET_REPO}' already exists.")
except RepositoryNotFoundError:
    print(f"Repository not found. Creating a new public dataset repository: '{DATASET_REPO}'...")
    create_repo(
        repo_id=DATASET_REPO,
        repo_type=REPO_TYPE,
        private=False,
        token=HF_TOKEN
    )
    print("Repository created successfully.")

# 2. Local directory structural check and asset tracking
local_data_dir = "tourism_project/data"
os.makedirs(local_data_dir, exist_ok=True)

# Define path configurations
local_csv_path = os.path.join(local_data_dir, "tourism.csv")

# If the file isn't in the project directory yet, move or read it
if not os.path.exists(local_csv_path):
    # Assuming 'tourism.csv' is present in your current working directory from the upload
    if os.path.exists("tourism.csv"):
        df = pd.read_csv("tourism.csv")
        df.to_csv(local_csv_path, index=False)
        print(f"Staged 'tourism.csv' into local pipeline directory: {local_data_dir}")
    else:
        raise FileNotFoundError("Could not find 'tourism.csv' in the working path to register. Please ensure it is uploaded.")

# 3. Synchronize local data artifacts into the cloud storage repository
print(f"Uploading data assets from '{local_data_dir}' to Hugging Face Dataset Hub...")
api.upload_folder(
    folder_path=local_data_dir,
    repo_id=DATASET_REPO,
    repo_type=REPO_TYPE,
    path_in_repo=""
)

print(f"Raw data registration complete! Accessible at: https://huggingface.co/datasets/{DATASET_REPO}")
