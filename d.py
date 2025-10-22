from huggingface_hub import snapshot_download
 
repo_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # hoặc mô hình bạn cần
local_dir = "./models/my_model"
snapshot_download(repo_id=repo_id, local_dir=local_dir)