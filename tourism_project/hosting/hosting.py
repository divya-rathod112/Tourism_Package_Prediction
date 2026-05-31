from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
SPACE_REPO = "divyarathod112/tourism-package-prediction" # <-- Replace with your HF username and Space name

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print("Deployment directory synced into Hugging Face Spaces Registry.")
