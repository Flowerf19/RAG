import sys
import os

# Fix Unicode output encoding for Windows PowerShell
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append('C:/Users/ENGUYEHWC/Prototype/Version_03/RAG')

from pipeline.rag_pipeline import RAGPipeline
from embedders.providers.ollama import OllamaModelType

def run_pipeline():
    print("🚀 CHẠY RAG PIPELINE - XỬ LÝ PDF VÀ TẠO EMBEDDINGS")
    print("=" * 60)

    try:
        # Khởi tạo pipeline với Gemma embedder
        pipeline = RAGPipeline(
            output_dir="data",
            model_type=OllamaModelType.GEMMA
        )

        print("✅ Pipeline đã khởi tạo thành công")
        print("📁 Đang xử lý tất cả PDF trong thư mục data/pdf...")

        # Xử lý tất cả PDF trong thư mục
        results = pipeline.process_directory()

        print(f"\n✅ HOÀN THÀNH! Đã xử lý {len(results)} PDF")
        print("\n📊 KẾT QUẢ CHI TIẾT:")

        for i, result in enumerate(results, 1):
            print(f"\n--- PDF {i}: {result.get('file_name', 'Unknown')} ---")
            
            if result.get('success') is False:
                print(f"❌ Lỗi: {result.get('error', 'Unknown error')}")
                continue
            
            print(f"📄 Số trang: {result.get('pages', 0)}")
            print(f"✂️ Số chunks: {result.get('chunks', 0)}")
            print(f"🧠 Embeddings: {result.get('embeddings', 0)}")
            print(f"⏭️  Chunks đã xử lý trước: {result.get('skipped_chunks', 0)}")
            print(f"� Dimension: {result.get('dimension', 0)}")
            
            files = result.get('files', {})
            if files:
                print(f"💾 FAISS Index: ✅ {files.get('faiss_index', 'N/A').split('/')[-1]}")
                print(f"📋 Metadata Map: ✅ {files.get('metadata_map', 'N/A').split('/')[-1]}")
                print(f"📄 Summary: ✅ {files.get('summary', 'N/A').split('/')[-1]}")

        print("\n" + "="*60)
        print("🎉 PIPELINE HOÀN THÀNH! Dữ liệu đã được lưu vào thư mục data/")

    except Exception as e:
        print(f"❌ LỖI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()