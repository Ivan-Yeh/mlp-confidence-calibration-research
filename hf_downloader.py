import sys
from huggingface_hub import snapshot_download, login

try:
    _, token, model = sys.argv
    login(token=token)
    print(f"Downloading: {model}")
    local_dir = snapshot_download(repo_id=model)
    print(f"Model saved to: {local_dir}")
except Exception as e:
    print(e)