from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "narasi80/tourism_package_prediction"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.getenv("HF_TOKEN"))
    print(f"Repo '{repo_id}' created.")

# Step 2: Upload data folder
api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)



