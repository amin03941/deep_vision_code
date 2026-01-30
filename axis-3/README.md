# Microbione 🧬

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Latest-green.svg)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Multimodal Retrieval-Augmented Generation (RAG) system for microbiome data analysis**

Microbione is an advanced RAG system that processes clinical, genomic, and visualization data to enable cross-modal search, variant prioritization, therapeutic analysis, and knowledge graph construction for microbiome research.

## ✨ Features

- 🔍 **Cross-Modal Search**: Search across text, sequence, and image embeddings simultaneously
- 🎯 **Variant Prioritizer**: Rank samples by genomic and clinical similarity
- 🔬 **Therapeutic Analysis**: Analyze treatment profiles and outcomes
- 📊 **Cluster Explorer**: Discover patterns through multi-modal clustering
- 🕸️ **Knowledge Graph**: Build relationship graphs from vector similarity and metadata
- 🎛️ **Advanced Filtering**: Filter by clinical parameters (BMI, IRIS, FPG, age, gender, etc.)

## Quick Start

### 1. Start Qdrant

```bash
docker-compose up -d
```

### 2. Import Collections

```bash
python quick_import.py
```

This imports:
- `microbiome_text` - 2,901 text vectors (384-dim)
- `microbiome_sequence` - 2,901 sequence vectors (384-dim)
- `microbiome_image` - 5,802 image vectors (512-dim)

### 3. Query Your Data

```python
from query_rag import MicrobiomeRAG

rag = MicrobiomeRAG()

# Cross-modal search
results = rag.cross_modal_search("diabetic patient with high BMI", limit_per_modality=3)

for result in results:
    print(f"Score: {result['score']:.4f} | Type: {result['type']}")
    print(f"Sample: {result['payload']['sample_id']}")
    print()
```

Or test it:
```bash
python test_rag.py
```

## Project Structure

```
.
├── import_all_collections_to_qdrant.py  # Import script
├── quick_import.py                       # Quick import wrapper
├── query_rag.py                          # RAG query interface
├── test_rag.py                           # Test script
├── docker-compose.yml                    # Qdrant setup
├── requirements.txt                       # Dependencies
├── qdrant_*.pkl                          # Vector data files
└── qdrant_storage/                       # Qdrant data directory
```

## Collections

| Collection | Vectors | Dimensions | Description |
|------------|---------|------------|-------------|
| `microbiome_text` | 2,901 | 384 | Clinical + genomic text descriptions |
| `microbiome_sequence` | 2,901 | 384 | 16S rRNA sequence embeddings |
| `microbiome_image` | 5,802 | 512 | Visualization embeddings |

## API Usage

```python
from query_rag import MicrobiomeRAG

rag = MicrobiomeRAG()

# Text search
results = rag.search_text("diabetic patient", limit=5)

# Sequence search
results = rag.search_sequence("stool microbiome", limit=5)

# Image search
results = rag.search_image("clinical markers", limit=5)

# Cross-modal search (all collections)
results = rag.cross_modal_search("prediabetic insulin resistant", limit_per_modality=3)
```

## 📋 Requirements

- Python 3.8+
- Docker (for Qdrant)
- See `requirements.txt` for Python packages

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/josephsenior/Microbione.git
cd Microbione

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker-compose up -d

# Import collections
python quick_import.py
```

## 📊 Dataset

The system processes the **MPEG-G Microbiome Classification Challenge** dataset:
- **2,901 samples** with clinical metadata
- **16S rRNA sequences** (FastQ files)
- **Visualization data** (heatmaps, charts, statistical plots)

**Total Vectors Indexed**: 11,604
- Text: 2,901 vectors (384-dim)
- Sequence: 2,901 vectors (384-dim)
- Image: 5,802 vectors (512-dim)

## Advanced Features

### Variant Prioritizer

Rank samples by genomic and clinical similarity:

```python
from advanced_features import AdvancedMicrobiomeRAG

rag = AdvancedMicrobiomeRAG()

# Prioritize variants based on reference
prioritized = rag.variant_prioritizer(
    reference_query="diabetic patient with high BMI, insulin resistant",
    clinical_filters={'class': 'Diabetic', 'bmi_min': 25},
    prioritize_by="sequence",  # or "text" or "combined"
    limit=10
)

for result in prioritized:
    print(f"Sample: {result['payload']['sample_id']}")
    print(f"Composite Score: {result['composite_score']:.3f}")
    print(f"Similarity: {result['similarity_score']:.3f}")
    print(f"Clinical: {result['clinical_score']:.3f}")
```

### Advanced Filtering

Filter by clinical parameters:

```python
# Filter by multiple criteria
results = rag.advanced_filter(
    query="prediabetic patient",
    filters={
        'class': 'Prediabetic',
        'iris': 'IS',
        'bmi_min': 25,
        'bmi_max': 35,
        'age_min': 40,
        'gender': 'F'
    },
    modality="text",
    limit=10
)
```

### Therapeutic Analysis

Analyze treatment profiles and outcomes:

```python
analysis = rag.therapeutic_analysis(
    treatment_profile="insulin sensitive diabetic patient",
    outcome_filter={'iris': 'IS'},
    limit=10
)

print(f"Total matches: {analysis['total_matches']}")
print(f"Statistics: {analysis['statistics']}")
print(f"Recommendations: {analysis['recommendations']}")
```

### Cluster Explorer

Discover clusters of similar samples:

```python
clusters = rag.cluster_explorer(
    query="diabetic samples",
    n_clusters=5,
    modality="text",
    min_cluster_size=3
)

for cluster_id, cluster_info in clusters['clusters'].items():
    print(f"Cluster {cluster_id}: {cluster_info['size']} samples")
    print(f"Characteristics: {cluster_info['characteristics']}")
```

### Knowledge Graph

Build and explore knowledge graphs:

```python
from knowledge_graph import MicrobiomeKnowledgeGraph

kg = MicrobiomeKnowledgeGraph(rag)

# Build graph
graph = kg.build_graph(
    similarity_threshold=0.7,
    include_metadata_edges=True
)

# Get statistics
stats = kg.get_statistics()

# Find central samples
central = kg.get_central_samples(top_n=10)

# Find communities
communities = kg.find_communities()

# Export for visualization
kg.export_graph("microbiome_graph.json")
```

## Testing

Test all features:

```bash
# Test basic RAG
python test_rag.py

# Test advanced features
python test_advanced_features.py
```

## 📁 Project Structure

```
Microbione/
├── query_rag.py                    # Basic RAG interface
├── advanced_features.py             # Advanced features (prioritizer, filtering, etc.)
├── knowledge_graph.py               # Knowledge graph builder
├── import_all_collections_to_qdrant.py  # Import script
├── quick_import.py                  # Quick import wrapper
├── test_rag.py                     # Basic tests
├── test_advanced_features.py       # Advanced feature tests
├── docker-compose.yml              # Qdrant setup
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Qdrant](https://qdrant.tech/) for vector database
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [CLIP](https://openai.com/research/clip) for image embeddings
- MPEG-G Microbiome Classification Challenge dataset
