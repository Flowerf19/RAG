"""
Test chunker with new config loading
"""

from chunkers.semantic_chunker import SemanticChunker
from chunkers.config_loader import load_chunker_config

def test_chunker_config():
    """Test config loading and chunker initialization"""
    
    print("🔧 Testing Chunker Config Loading")
    print("=" * 50)
    
    # Test config loading
    try:
        config = load_chunker_config()
        print(f"✅ Config loaded successfully")
        print(f"Available chunker types: {list(config.keys())}")
        
        # Check semantic config
        semantic_config = config.get('semantic', {})
        print(f"\n📊 Semantic Chunker Config:")
        print(f"  max_tokens: {semantic_config.get('max_tokens')}")
        print(f"  overlap_tokens: {semantic_config.get('overlap_tokens')}")
        print(f"  similarity_threshold: {semantic_config.get('similarity_threshold')}")
        print(f"  min_chunk_size: {semantic_config.get('min_chunk_size')}")
        print(f"  default_language: {semantic_config.get('default_language')}")
        
        spacy_models = semantic_config.get('spacy_models', {})
        print(f"  spaCy models: {len(spacy_models)} languages")
        for lang, model in list(spacy_models.items())[:5]:  # Show first 5
            print(f"    {lang}: {model}")
        if len(spacy_models) > 5:
            print(f"    ... and {len(spacy_models)-5} more")
            
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return
    
    print(f"\n🧩 Testing Chunker Initialization")
    print("=" * 50)
    
    # Test chunker with config
    try:
        chunker = SemanticChunker()
        print(f"✅ Chunker initialized from config")
        print(f"  max_tokens: {chunker.max_tokens}")
        print(f"  overlap_tokens: {chunker.overlap_tokens}")
        print(f"  language: {chunker.language}")
        print(f"  similarity_threshold: {chunker.similarity_threshold}")
        print(f"  discourse_weight: {chunker.discourse_weight}")
        print(f"  lexical_weight: {chunker.lexical_weight}")
        print(f"  entity_weight: {chunker.entity_weight}")
        
    except Exception as e:
        print(f"❌ Chunker initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test chunker with overrides
    try:
        chunker_override = SemanticChunker(max_tokens=300, language="vi")
        print(f"\n✅ Chunker with overrides initialized")
        print(f"  max_tokens: {chunker_override.max_tokens} (overridden)")
        print(f"  overlap_tokens: {chunker_override.overlap_tokens} (from config)")
        print(f"  language: {chunker_override.language} (overridden)")
        
    except Exception as e:
        print(f"❌ Chunker override failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chunker_config()