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

# Target configuration
HF_USERNAME = "divyarathod112" 
DATASET_REPO = f"{HF_USERNAME}/Tourism-Package-Prediction"
REPO_TYPE = "dataset"

# 1. Ensure the remote data repository exists on Hugging Face Hub
try:
    api.repo_info(repo_id=DATASET_REPO, repo_type=REPO_TYPE)
    print(f"✅ Dataset repository '{DATASET_REPO}' already exists.")
except RepositoryNotFoundError:
    print(f"📁 Repository not found. Creating a new public dataset repository: '{DATASET_REPO}'...")
    create_repo(repo_id=DATASET_REPO, repo_type=REPO_TYPE, private=False, token=HF_TOKEN)
    print("✅ Repository created successfully.")

# 2. Establish and verify local data staging layout
local_data_dir = "tourism_project/data"
os.makedirs(local_data_dir, exist_ok=True)
local_csv_path = os.path.join(local_data_dir, "tourism.csv")

# 3. Locate tourism.csv dynamically in GitHub Runner environments
if not os.path.exists(local_csv_path):
    # Search locations (current directory, parent directory, or nested)
    possible_paths = ["tourism.csv", "../tourism.csv", "tourism_project/tourism.csv"]
    found_path = None
    
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if found_path:
        df = pd.read_csv(found_path)
        df.to_csv(local_csv_path, index=False)
        print(f"📦 Successfully staged dataset from '{found_path}' into '{local_csv_path}'")
    else:
        # Debug helper: print out files present in the runner path to see what's visible
        print(f"Current Directory Contents: {os.listdir('.')}")
        raise FileNotFoundError("Could not find 'tourism.csv' in the working path to register. Please ensure it is tracked in your repository or uploaded.")

# 4. Synchronize data assets to Hugging Face
print(f"🚀 Uploading data assets from '{local_data_dir}' to Hugging Face...")
api.upload_folder(
    folder_path=local_data_dir,
    repo_id=DATASET_REPO,
    repo_type=REPO_TYPE,
    path_in_repo=""
)
print(f"🎉 Raw data registration complete! Hub URL: https://huggingface.co/datasets/{DATASET_REPO}")
