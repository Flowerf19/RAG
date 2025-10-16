import sys
import os
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
            print(f"📄 Số trang: {result.get('pages_processed', 0)}")
            print(f"✂️ Số chunks: {result.get('chunks_created', 0)}")
            print(f"🧠 Embeddings: {result.get('embeddings_created', 0)}")
            print(f"💾 Vector index: {'✅' if result.get('vector_index_saved') else '❌'}")
            print(f"📋 Metadata: {'✅' if result.get('metadata_saved') else '❌'}")

            if result.get('errors'):
                print(f"⚠️ Lỗi: {result['errors']}")

        print("\n" + "="*60)
        print("🎉 PIPELINE HOÀN THÀNH! Dữ liệu đã được lưu vào thư mục data/")

    except Exception as e:
        print(f"❌ LỖI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()