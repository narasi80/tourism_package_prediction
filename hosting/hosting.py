from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

repo_id = "Narasi/tourism_package_prediction_space"
repo_type = "space"

# Check if space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN, space_sdk="gradio")
    print(f"Space '{repo_id}' created.")

# Upload deployment folder using relative path
api.upload_folder(
    folder_path="deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
)
print("Deployment files uploaded successfully!")
