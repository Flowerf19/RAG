#!/usr/bin/env python3
"""
Test script để verify HuggingFace API token hoạt động
"""

import os
import requests
import sys

def test_hf_token():
    """Test HuggingFace API token"""
    
    # Check environment variables
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    
    if not hf_token:
        print("❌ Không tìm thấy HF_TOKEN hoặc HUGGINGFACE_TOKEN")
        print("💡 Thiết lập token bằng:")
        print("   export HF_TOKEN='your_token_here'")
        return False
    
    print(f"✅ Tìm thấy token: {hf_token[:10]}...")
    
    # Test API call
    try:
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = requests.get(
            "https://huggingface.co/api/whoami-v2", 
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_data = response.json()
            print(f"✅ Token hợp lệ! User: {user_data.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Token không hợp lệ. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        return False

def test_hf_embedding_api():
    """Test HuggingFace Inference API cho embedding"""
    
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    
    if not hf_token:
        print("❌ Cần HF_TOKEN để test embedding API")
        return False
    
    try:
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": "Hello world",
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                embedding = result[0]
                print(f"✅ Embedding API hoạt động! Vector size: {len(embedding)}")
                return True
        
        print(f"❌ Embedding API lỗi. Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        return False
        
    except Exception as e:
        print(f"❌ Lỗi test embedding: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing HuggingFace API Token\n")
    
    # Test 1: Token validity
    print("1. Kiểm tra token hợp lệ:")
    token_ok = test_hf_token()
    print()
    
    # Test 2: Embedding API
    print("2. Test Embedding API:")
    if token_ok:
        api_ok = test_hf_embedding_api()
        if api_ok:
            print("\n🎉 Tất cả test thành công! HF API sẵn sàng sử dụng.")
        else:
            print("\n⚠️ Token hợp lệ nhưng API có thể cần thời gian để load model.")
    else:
        print("❌ Bỏ qua test API vì token không hợp lệ.")
