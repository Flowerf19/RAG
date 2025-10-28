# from huggingface_hub import snapshot_download

# repo_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # hoặc mô hình bạn cần
# local_dir = "/workspaces/RAG/test/model"
# snapshot_download(repo_id=repo_id, local_dir=local_dir)


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = r"D:\intro Notelm and rad\ChatBot\RAG\rerank_model\model"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

query = "City in Germany"
docs = ["Paris is the capital of France.", "Berlin is the capital of Germany.", "Tokyo is in Japan."]

scores = []
for doc in docs:
    inputs = tokenizer(query, doc, return_tensors="pt", truncation=True)
    with torch.no_grad():
        score = model(**inputs).logits[0].item()
    scores.append((doc, score))

# Sắp xếp theo điểm cao nhất
ranked = sorted(scores, key=lambda x: x[1], reverse=True)

for doc, score in ranked:
    sigmoid_score = torch.sigmoid(torch.tensor(score)).item()
    print(f"{sigmoid_score:.4f} → {doc}")
inputs = tokenizer("City in Germany", "Berlin is the capital of Germany.", return_tensors="pt")
outputs = model(**inputs)

print(outputs.__class__)     # kiểu output
print(outputs.keys())        # các trường có trong output
print(outputs.logits)        # giá trị chính (score)