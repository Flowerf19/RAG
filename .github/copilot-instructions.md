# RAG System – Copilot Development Instructions

## 🧠 Overview
This repository implements a modular **Retrieval-Augmented Generation (RAG)** system designed with **Object-Oriented Programming (OOP)** principles.

The pipeline includes:
1. **Loader Module** → Handles PDF/DOCX ingestion and normalization. ✅ *(Completed)*
2. **Chunker Module** → Handles text segmentation into embedding-ready chunks. 🚧 *(Current focus)*

Goal: Transform PDF/DOCX → `NormalizedDocument` → `ChunkSet` → (Embedding → Reranking → Retrieval).

---

## ⚙️ Environment Setup
```powershell
# Activate virtual environment
& C:/Users/ENGUYEHWC/Downloads/RAG/RAG/.venv/Scripts/Activate.ps1

# Install dependencies
pip install -r requirements.txt
📁 Project Structure
bash
RAG/
├── .venv/                         # Virtual environment
├── loaders/                       # ✅ PDF/DOCX ingestion
│   ├── pdf_loader.py
│   ├── model/
│   └── normalizers/
├── chunkers/                      # 🚧 Current focus
│   ├── hybrid_chunker.py          # Main orchestrator
│   ├── semantic_chunker.py        # Semantic segmentation
│   ├── rule_chunker.py            # Rule-based segmentation
│   ├── fixed_chunker.py           # Fixed-length fallback
│   ├── model.py                   # Shared data classes
│   └── utils.py                   # Token estimator & helpers
├── tests/
│   ├── test_loader.py
│   └── test_chunker.py
└── copilot-instruc.md             # Copilot & developer guidance
📚 Loader Module Summary (✅ Completed)
File: loaders/pdf_loader.py
Purpose: Extract and normalize PDF/DOCX files.

Output schema:

python
Sao chép mã
NormalizedDocument {
  documentId: UUID,
  metadata: {...},
  blocks: [
    {
      blockId: UUID,
      type: "paragraph" | "heading" | "list" | "table" | "code",
      text: str,
      provenance: { file, page, charRange }
    }
  ]
}
Key Features

Dependency injection (config via constructor)

Text & table extraction

Config validation

Factory methods: create_default(), create_text_only(), create_tables_only()

OOP encapsulation and static utilities

🔧 Chunker Module (🚧 Current Focus)
🎯 Objective
Convert a normalized document into semantically meaningful chunks.
Implements Hybrid Chunking: combining semantic, rule-based, and fixed-size strategies.

🧩 Class Overview
Class	Responsibility
HybridChunker	Main orchestrator; selects and manages strategies
SemanticChunker	Semantic segmentation using text coherence or embeddings
RuleBasedChunker	Structural segmentation by headings, lists, tables
FixedSizeChunker	Token-length fallback segmentation
ChunkSet	Holds list of chunks for a document
Chunk	Represents one embedding-ready segment
ProvenanceAgg	Aggregates provenance from all contributing blocks
BlockSpan	Represents character offsets within source blocks
Score	Chunk quality metrics
ChunkStats	Aggregated chunking statistics

⚙️ Architecture Flow
css

NormalizedDocument
      ↓
 HybridChunker
 ├─ SemanticChunker
 ├─ RuleBasedChunker
 └─ FixedSizeChunker
      ↓
   ChunkSet
    └── [Chunk → ProvenanceAgg → BlockSpan]
🧠 HybridChunker Parameters
python
HybridChunker(
  targetTokens=200,
  minTokens=100,
  maxTokens=400,
  overlapRatio=0.1,
  language="en"
)
💡 Core Methods
Method	Description
HybridChunker.chunk(doc)	Entry point; orchestrates all strategies
HybridChunker.evaluateAndRefine(set)	Optional QA step
SemanticChunker.chunkSegment(blocks)	Splits by semantic boundaries
RuleBasedChunker.chunkByRules(blocks)	Splits by structural rules
FixedSizeChunker.chunkByLength(blocks)	Splits evenly by token length

🧱 Data Models (chunkers/model.py)
python
@dataclass
class BlockSpan:
    blockId: str
    charStart: int
    charEnd: int
    bbox: Optional[str] = None

@dataclass
class ProvenanceAgg:
    file: str
    sha256Doc: str
    pageRanges: List[int]
    blockSpans: List[BlockSpan]

@dataclass
class Score:
    cohesion: float = 0
    topicShift: float = 0
    structureConf: float = 0
    boundaryConf: float = 0

@dataclass
class Chunk:
    chunkId: str
    order: int
    textForEmbedding: str
    tokensEstimate: int
    contentType: str
    scores: Score
    provenance: ProvenanceAgg

@dataclass
class ChunkStats:
    numChunks: int
    avgTokens: int
    stdevTokens: float

@dataclass
class ChunkSet:
    documentId: str
    chunks: List[Chunk]
    stats: Optional[ChunkStats] = None