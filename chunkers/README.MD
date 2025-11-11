# Chunkers Module — Semantic Text Segmentation with Multi-Language Support

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

The chunkers module provides an intelligent text segmentation system using spaCy for sentence splitting and coherence scoring. It follows the Single Responsibility principle: chunkers only handle segmentation, not embedding. Supports multiple languages with automatic spaCy model selection. This is the second component in the RAG pipeline: `PDF → PDFProvider → PDFDocument → SemanticChunker → ChunkSet → ...`

## ✨ Key Features

- 🔍 **Semantic Segmentation**: Intelligent text chunking using spaCy sentence analysis
- 🌐 **Multi-Language Support**: 13+ languages with automatic model selection
- 📊 **Coherence Scoring**: Discourse markers, lexical overlap, and entity analysis
- 🧩 **Modular Architecture**: Factory patterns, composition design, dependency injection
- 📈 **Performance Optimized**: Memory-efficient processing with configurable parameters
- 🔄 **Provenance Tracking**: Source information preservation across chunks
- 📏 **Flexible Configuration**: Token limits, overlap settings, and scoring weights

## 🚀 Quick Start

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install spaCy models
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
python -c "import spacy; spacy.cli.download('vi_core_news_lg')"
```

### Basic Usage

The module provides intelligent text segmentation with automatic language detection and coherence-based chunking.

## 🔧 Component Design & Behavior

### SemanticChunker (semantic_chunker.py)

The main chunker uses spaCy for sentence segmentation and coherence analysis. It supports 13+ languages with automatic model selection and provides configurable parameters for chunk size, overlap, and scoring weights.

### BaseChunker (base_chunker.py)

Abstract base class defining the interface for all chunkers. It provides the foundation for implementing different chunking strategies while maintaining consistent behavior across implementations.

## 📊 Data Models

### Chunk
A dataclass representing an individual chunk with unique identifier, content text, chunk type, source information, coherence score, and token count.

### ChunkSet
A dataclass representing a collection of chunks from a document, including total counts, token statistics, and performance metrics.

### ProvenanceAgg
A dataclass tracking source information including filename, page numbers, text spans, and content types for traceability.

### Score
A dataclass containing coherence scores including overall score, discourse markers score, lexical overlap score, and entity overlap score.

## 💡 Usage Examples

### Basic Usage
Load a PDF document and create semantic chunks with automatic language detection and coherence scoring.

### Advanced Configuration
Specify language explicitly, adjust token limits, and customize coherence scoring parameters for different document types.

### Integration with Pipeline
The chunker integrates seamlessly with the RAG pipeline for automatic document processing.

## 🌐 Language Support & Setup

### Supported Languages
- **English** (`en`): en_core_web_sm
- **Vietnamese** (`vi`): vi_core_news_lg (large model recommended)
- **Chinese** (`zh`): zh_core_web_sm
- **European**: German, French, Spanish, Italian, Portuguese, Dutch, Polish
- **Russian** (`ru`): ru_core_news_sm
- **Japanese** (`ja`): ja_core_news_sm
- **Multilingual** (`multilingual`): xx_ent_wiki_sm

### Installing spaCy Models
Install the required spaCy language models for your documents.

### Language Auto-Detection
Automatically detects document language from content and selects the appropriate spaCy model.

## 🧪 Testing

### Unit Tests
Test individual chunking components, language detection, and coherence scoring algorithms.

### Integration Tests
Test complete workflows with sample documents to verify chunking quality and performance.

### Performance Testing
Measure chunking speed and quality metrics for different document types and sizes.

## 📦 Dependencies & Installation

### Core Dependencies
Essential libraries for NLP processing and text analysis.

### Installation
Install from requirements file and download necessary spaCy language models.

## ⚠️ Operational Notes

### Performance Considerations
spaCy models require significant memory and load time. Processing speed varies by model size and document length. Consider caching for repeated processing.

### Language-Specific Tuning
Different languages may require different model sizes and processing approaches for optimal results.

### Error Handling
Clear error messages guide users when models are missing or documents cannot be processed.

### Best Practices
Match chunker language to document language, adjust token limits based on embedder constraints, and tune overlap for coherence.

## 🔧 Troubleshooting

### Common Issues

**spaCy Model Not Found:**
Check installed models and install missing ones as needed.

**Poor Chunk Quality:**
Use specific language settings and examine coherence scores to improve chunking quality.

**Language Detection Issues:**
Force specific language settings when auto-detection fails.

**Memory Issues:**
Process large documents in batches or use streaming approaches.

### Debug Mode
Enable detailed logging to troubleshoot chunking behavior and performance.

## API Reference

### SemanticChunker
- `chunk_document(pdf_document) → ChunkSet`
- `chunk_pages(pages) → ChunkSet`
- `_aggregate_page_content(page) → str`
- `_detect_language(text) → str`
- `get_chunk_stats() → ChunkStats`

### BaseChunker (Abstract)
- `chunk_document(pdf_document) → ChunkSet` — *abstract*
- `get_chunk_stats() → ChunkStats` — *abstract*
- `_validate_inputs(pdf_document)` — *abstract*

### Data Models
- `Chunk` — Individual chunk with metadata
- `ChunkSet` — Collection of chunks
- `ChunkStats` — Performance metrics
- `ProvenanceAgg` — Source tracking
- `Score` — Coherence scores

## 🤝 Contributing

### Adding New Languages
1. Add language code to `SPACY_MODEL_MAP`
2. Install and test spaCy model
3. Update language detection logic if needed
4. Add tests for new language
5. Update documentation

### Custom Scoring Algorithms
1. Extend `Score` dataclass
2. Implement scoring method in chunker
3. Add configuration parameters
4. Test scoring accuracy
5. Update benchmarks

### Performance Optimizations
1. Profile current implementation
2. Identify bottlenecks (model loading, text processing)
3. Implement caching strategies
4. Add batch processing support
5. Benchmark improvements</content>
<parameter name="filePath">d:\Project\RAG-2\chunkers\README.MD