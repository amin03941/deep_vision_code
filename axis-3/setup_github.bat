@echo off
REM Setup script for GitHub repository (Windows)

echo Setting up GitHub repository...

REM Push to GitHub
echo Pushing code to GitHub...
git push -u origin main

REM Set repository description and topics using GitHub CLI
echo Setting repository description and topics...

gh repo edit josephsenior/Microbione --description "Multimodal RAG system for microbiome data analysis with cross-modal search, variant prioritization, and knowledge graph construction" --add-topic "rag" --add-topic "microbiome" --add-topic "vector-database" --add-topic "qdrant" --add-topic "bioinformatics" --add-topic "machine-learning" --add-topic "embeddings" --add-topic "cross-modal-search" --add-topic "knowledge-graph" --add-topic "genomics" --add-topic "clinical-data" --add-topic "python"

echo.
echo Repository setup complete!
echo Visit: https://github.com/josephsenior/Microbione
pause
