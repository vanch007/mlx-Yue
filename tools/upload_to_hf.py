"""Upload converted MLX weights to Hugging Face Hub."""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login


def main():
    parser = argparse.ArgumentParser(description="Upload mlx-Yue models to Hugging Face")
    parser.add_argument("--repo-id", default="vanch007/mlx-Yue2-3B", help="Target Hugging Face repo ID")
    parser.add_argument("--folder", default="models/converted", type=Path, help="Folder containing weights and README")
    parser.add_argument("--token", default=None, help="Hugging Face write token (defaults to HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create as private repository")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=True)
    
    api = HfApi(token=token)
    try:
        user = api.whoami()
        print(f"Logged in as Hugging Face user: {user.get('name', user.get('user', 'unknown'))}")
    except Exception as e:
        print(f"Error: Unable to authenticate with Hugging Face: {e}", file=sys.stderr)
        print("Please provide a valid token with write access via --token or HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    print(f"Creating/verifying repository: {args.repo_id}...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    print(f"Uploading files from {args.folder} to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(args.folder),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add native MLX YuE2-3B weights (BF16 & 8-bit quantized) and Model Card",
    )
    print(f"\nUpload complete! Model URL: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

