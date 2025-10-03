# TODO List - Chunkers Module

## Core Implementation

### High Priority

- [ ] Implement complete logic cho `FixedSizeChunker.chunk()`
- [ ] Implement complete logic cho `SemanticChunker.chunk()`
- [ ] Implement complete logic cho `SlidingWindowChunker.chunk()`
- [ ] Implement `HierarchicalChunker` strategy
- [ ] Add token counting support với tiktoken/transformers
- [ ] Integrate với loaders module (PDFDocument, PDFPage, Block, Table)

### Medium Priority

- [ ] Add sentence boundary detection (spacy/nltk)
- [ ] Add paragraph detection logic
- [ ] Add section/heading detection
- [ ] Implement table chunking strategies (separate/inline/hybrid)
- [ ] Add overlap handling cho fixed size và sliding window
- [ ] Implement word boundary preservation
- [ ] Add support cho multi-page chunks

### Low Priority

- [ ] Add hierarchical chunking với parent-child relationships
- [ ] Add chunk linking (previous/next chunk IDs)
- [ ] Add embedding generation support
- [ ] Add chunk quality metrics
- [ ] Add chunk deduplication logic

## Processors

- [ ] Enhance `TextProcessor` với advanced text normalization
- [ ] Add `HybridProcessor` cho mixed text+table content
- [ ] Add markdown formatting utilities
- [ ] Add custom tokenizers support

## Configuration

- [ ] Add validation cho config parameters
- [ ] Add config presets cho common use cases
- [ ] Add runtime config override support

## Testing

### Unit Tests

- [ ] Test `Chunk` model validation
- [ ] Test `ChunkMetadata` creation
- [ ] Test `ChunkDocument` statistics
- [ ] Test `FixedSizeStrategy` splitting
- [ ] Test `SemanticStrategy` splitting
- [ ] Test `SlidingWindowStrategy` splitting
- [ ] Test `TextProcessor` normalization
- [ ] Test `TableProcessor` serialization
- [ ] Test ID generation (stable, deterministic)
- [ ] Test citation generation

### Integration Tests

- [ ] Test chunking với real PDF documents từ loaders
- [ ] Test end-to-end pipeline (load -> normalize -> chunk)
- [ ] Test với various document types
- [ ] Test với large documents (performance)
- [ ] Test với tables, images, mixed content

### Quality Tests

- [ ] Verify chunk quality (coherence, completeness)
- [ ] Verify no content loss during chunking
- [ ] Verify overlap is working correctly
- [ ] Verify metadata accuracy

## Documentation

- [ ] Add docstrings cho all public methods
- [ ] Add usage examples
- [ ] Add chunking strategy comparison guide
- [ ] Add configuration guide
- [ ] Add troubleshooting guide
- [ ] Add performance tuning guide

## Performance

- [ ] Profile chunking performance
- [ ] Optimize large document handling
- [ ] Add caching cho repeated operations
- [ ] Add parallel processing support (multi-threading/processing)
- [ ] Optimize memory usage

## Features

- [ ] Add support cho custom chunking strategies
- [ ] Add chunk merging/splitting utilities
- [ ] Add chunk filtering by criteria
- [ ] Add chunk ranking/scoring
- [ ] Add chunk summarization
- [ ] Add chunk visualization tools

## Integration

- [ ] Ensure compatibility với loaders module
- [ ] Add export formats (JSON, CSV, Parquet)
- [ ] Add integration với vector databases
- [ ] Add integration với embedding models
- [ ] Add CLI interface

## Error Handling

- [ ] Add comprehensive error handling
- [ ] Add validation errors với helpful messages
- [ ] Add logging throughout
- [ ] Add debugging utilities
- [ ] Add error recovery mechanisms

## Edge Cases

- [ ] Handle empty documents
- [ ] Handle very small documents (< min chunk size)
- [ ] Handle very large documents (> memory limits)
- [ ] Handle documents với no structure
- [ ] Handle documents với complex layouts
- [ ] Handle multi-language documents
- [ ] Handle special characters, unicode
- [ ] Handle corrupted or malformed data

## Notes

- Prioritize integration với loaders module first
- Focus on semantic chunking as default strategy
- Ensure deterministic IDs for reproducibility
- Keep API simple and extensible
- Document all configuration options clearly
