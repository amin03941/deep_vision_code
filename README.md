\# 🧬 Bio-Scout: Multimodal Biological Design \& Discovery Intelligence

\#data link : https://drive.google.com/drive/folders/1aGkNzSGp8K-7DSiDeNr5tjSPem9MdFyG?usp=sharing

Bio-Scout is a multimodal biological intelligence system integrating genomic data (FASTQ 16S rRNA sequences), clinical metadata, and semantic embeddings into a unified vector space for personalized therapeutic discovery.



---



\## 📁 Repository Structure



```

bio-scout/

├── axe1/                          # Genetic Neighbor Finder

│   ├── config.py

│   ├── main.py

│   ├── evaluate.py

│   └── scripts/

│       ├── 1\_extract\_and\_parse.py

│       ├── 2\_vectorize.py

│       ├── 3\_index\_qdrant.py

│       ├── 4\_search\_neighbors.py

│       └── 5\_therapeutic\_analysis.py

│

├── axe2/                          # Discovery Dashboard

│   ├── supcomhackathon.ipynb     # Kaggle notebook (DNABERT-2 vectorization)

│   ├── app.py                     # Streamlit dashboard

│   └── vector.py                  # Health direction vector generation

│

└── README.md

```



---



\## 🔬 Axe 1: Genetic Neighbor Finder + Therapeutic Analysis



\### Objective

For a given patient (SubjectID), identify the \*\*k most similar neighbors\*\* (genetically + clinically) and automatically generate a \*\*structured therapeutic report\*\* via LLM (Gemini Flash 2.5).



\### Input Data



\*\*Required Input\*\*: `subjectid.zip` containing 66 subjects structured as follows:



```

subjectid.zip

├── Subject\_UDAXIH/

│   ├── clinical.csv               # Clinical metadata (1 row)

│   └── fastq/

│       ├── Sample\_XXX.fastq       # Sequence 1

│       └── Sample\_YYY.fastq       # Sequence 2

├── Subject\_NHOSIZ/

│   ├── clinical.csv

│   └── fastq/

│       └── ...

└── ... (66 subjects total)

```



\*\*clinical.csv format\*\*:

```csv

SubjectID,FPG\_Mean,Class,Gender,BMI,OGTT,Adj.age,...

Subject\_UDAXIH,1.274,Diabetic,M,21.47,2.245,59.48,...

```



\### Installation



```bash

cd axe1

pip install biopython qdrant-client numpy pandas scikit-learn google-generativeai

```



\### Configuration



\*\*1. Place your data\*\*:

\- Put `subjectid.zip` in `data/subjectid.zip` (or update `ZIP\_PATH` in `config.py`)



\*\*2. Set up Gemini API Key\*\*:



Get your API key from \[Google AI Studio](https://aistudio.google.com/app/apikey)



\#### Linux/Mac:

```bash

export GEMINI\_API\_KEY="your\_key\_here"

python main.py

```



\#### Windows PowerShell:

```powershell

$env:GEMINI\_API\_KEY="your\_key\_here"

python main.py

```



\#### Windows CMD:

```cmd

set GEMINI\_API\_KEY=your\_key\_here

python main.py

```



\*\*Note\*\*: The API key is read from environment variable `GEMINI\_API\_KEY` (not stored in code).



\### Usage



\#### Run Complete Pipeline

```bash

python main.py

```



\*\*Pipeline steps\*\*:

1\. ✅ Extract and parse FASTQ files from ZIP

2\. ✅ Generate multimodal vectors (k-mers 500D + clinical 9D = 509D)

3\. ✅ Index 65 subjects in Qdrant (cosine distance)

4\. ✅ Interactive mode: analyze any patient



\*\*Example interaction\*\*:

```

Enter SubjectID to analyze (or 'quit'): UDAXIH



============================================================

📊 PATIENT ANALYSIS: Subject\_UDAXIH

============================================================



👤 Patient Profile:

&nbsp; • Class: Diabetic

&nbsp; • FPG Mean: 1.274

&nbsp; • BMI: 21.47

&nbsp; • Age: 59.48



🎯 Top 7 Similar Neighbors:

1\. Subject\_BHBZKM - Similarity: 0.901 - Class: Prediabetic

...



📈 Class Distribution:

&nbsp; • Prediabetic: 6 (85.7%)

&nbsp; • Crossover: 1 (14.3%)



🎯 Risk Assessment: ⚠️ HIGH RISK (85.7% sick neighbors)



🤖 Generating therapeutic report with Gemini...

💾 Report saved: reports/report\_Subject\_UDAXIH\_20260126\_175144.txt

```



\#### Run Validation (Leave-One-Out)

```bash

python evaluate.py

```



\*\*Output\*\*:

```

📊 LEAVE-ONE-OUT EVALUATION (k=7)

🔄 Evaluating 65 subjects...



📈 RESULTS:

&nbsp; • Accuracy: 0.662 (66.2%)

&nbsp; • F1-Score (weighted): 0.601



📋 Per-Class Performance:

&nbsp; • Prediabetic: 39/42 correct (92.9% recall) ✅

&nbsp; • Control: 2/11 correct (18.2% recall)

&nbsp; • Crossover: 2/10 correct (20.0% recall)

&nbsp; • Diabetic: 0/2 correct (0.0% recall)

```



\### Generated Outputs



\- `processed\_subjects.json` - Parsed data

\- `vectors.pkl` - Multimodal vectors + scaler

\- `qdrant\_data/` - Local vector database

\- `reports/report\_SubjectID\_\*.txt` - Therapeutic reports



\*\*Sample Report Content\*\*:

```

THERAPEUTIC REPORT - Subject\_UDAXIH

Generated: 2026-01-26 17:51:44



\## 1. Similarity Analysis

Patient Subject\_UDAXIH, classified as Diabetic with elevated FPG (1.27),

shows 85.7% of neighbors as Prediabetic/Crossover...



\## 2. Microbiome Hypotheses (Cautious)

\- Reduced microbial diversity (associated with elevated FPG)

\- Altered SCFA production (likely butyrate decrease)

...



\## 3. Therapeutic Recommendations

1\. High-fiber diet + polyphenols

&nbsp;  Rationale: Promotes SCFA production, modulates inflammation

&nbsp;  

2\. Mediterranean diet

&nbsp;  Rationale: Associated with better glucose management...

```



---



\## 🗺️ Axe 2: Discovery Dashboard + Vector Steering



\### Objective

Simulate \*in-silico\* therapeutic interventions by navigating the vector space toward healthier profiles using a \*\*health direction vector\*\* ($\\mathbf{d}\_{health}$).



---



\### Part 1: Data Processing (Kaggle)

Due to the heavy computational requirements of \*\*DNABERT-2\*\*, the vectorization step is performed on Kaggle using a \*\*T4 GPU\*\*.



1\. \*\*Upload\*\* the `supcomhackathon.ipynb` notebook to Kaggle.

2\. \*\*Ensure\*\* the following datasets are attached to your Kaggle environment:

&nbsp;   \* `malloulifares/d2d-cytokine-data`

&nbsp;   \* `trainmpeg` (`Train.csv`, `Train\_Subjects.csv`)

&nbsp;   \* `mpeg-g-microbiomeclassificationconvertedfastqfiles`

&nbsp;   \* `secondbatchoffastqfiles`

3\. \*\*Run the notebook\*\*: The notebook will process the 1,982 samples, generate hybrid embeddings, and compile them into a pickle file.

4\. \*\*Download the Output\*\*: Once complete, download `bio\_memory\_dump.pkl` from the Kaggle output directory to your local project folder.



---



\### Part 2: Local Environment Setup

Ensure you have \*\*Python 3.10+\*\* installed on your local machine.



\#### 1. Install Dependencies

Open your terminal and run:

```bash

cd axe2

pip install streamlit qdrant-client plotly networkx streamlit-agraph scikit-learn pandas numpy

```



\#### 2. Prepare the Workspace

Place the downloaded `bio\_memory\_dump.pkl` into the same directory as your Python scripts (`app.py` and `vector.py`).



---



\### Part 3: Generate the Health Direction Vector

Before launching the dashboard, you need to calculate the vector steering trajectory.



Run the vector generation script:

```bash

python vector.py

```

\*\*Output\*\*: This will generate `health\_direction\_vector.npy` and `vector\_metadata.pkl` based on a composite inflammatory index (TNFA, IL-22, EGF).



---



\### Part 4: Launch the Bio-Scout Dashboard

Start the Streamlit application:

```bash

streamlit run app.py

```



---



\### 🔬 Dashboard Features

The Streamlit dashboard offers 5 distinct analytical modules:



\*   \*\*🔬 Discovery Dashboard\*\*: The core search engine. Input a Sample ID to find biologically similar samples. Enable the \*In-Silico Treatment Simulator\* to apply the health direction vector and predict how reducing inflammation shifts the microbiome profile.

\*   \*\*🗺️ Vector Space Map\*\*: An interactive t-SNE projection of the 768-dimensional genomic embeddings. Color by Body Site, TNFA levels, or K-Means clusters to identify distinct phenotypic groupings.

\*   \*\*🕸️ Graph Explorer (GraphRAG)\*\*: A network science tool to navigate relationships between Sample IDs, Body Sites, Insulin Sensitivity, and Cytokine levels using a multi-hop knowledge graph.

\*   \*\*📊 Batch Analysis\*\*: Compare cytokine profiles (TNFA, IL-22, EGF) across different body sites using interactive box plots and summary statistics.

\*   \*\*🛠️ System Diagnostics\*\*: Monitor the health of the Qdrant database, graph parameters, and the effectiveness score of the health steering vector.



---



\## 📊 Key Results



\### Axe 1 Validation

\- \*\*Accuracy\*\*: 66.2% (43/65 subjects correctly classified)

\- \*\*Prediabetic Recall\*\*: 92.9% (39/42) - Excellent detection of critical class

\- \*\*F1-Score (weighted)\*\*: 0.601



\### Axe 2 Vector Quality

\- \*\*Separation Score\*\*: 0.825

\- \*\*Effectiveness\*\*: 0.018

\- \*\*Method\*\*: PCA-centroid blend (70/30)

\- \*\*Healthy/Disease Samples\*\*: 398/397



---



\## 🚀 Citation



If you use Bio-Scout in your research, please cite:



```bibtex

@software{bioscout2026,

&nbsp; title = {Bio-Scout: Multimodal Biological Design \& Discovery Intelligence},

&nbsp; author = {Team Neural Nomads},

&nbsp; year = {2026},

&nbsp; note = {Track 4 - Vectors in Orbit Hackathon}

}

```



---



\## 📝 Notes



\- \*\*Axe 1\*\*: Works entirely locally after data extraction (no GPU required)

\- \*\*Axe 2\*\*: Requires Kaggle GPU (T4) for DNABERT-2 embedding generation

\- \*\*Axes 3-4\*\*: Cluster Explorer and Variant Prioritizer are currently in development

\- Full technical documentation available in the project report (PDF)



---



\## 🔗 Links



\- \*\*Kaggle Dataset\*\*: \[MPEG-G Microbiome Classification](https://www.kaggle.com/datasets/noob786/mpeg-g-microbiomeclassificationconvertedfastqfiles),(https://www.kaggle.com/datasets/noob786/secondbatchoffastqfiles?select=TrainFiles),(https://www.kaggle.com/datasets/noob786/extrafiles)

\- \*\*DNABERT-2 Model\*\*: \[HuggingFace](https://huggingface.co/zhihan1996/DNABERT-2-117M)

\- \*\*Qdrant Documentation\*\*: \[https://qdrant.tech/documentation/](https://qdrant.tech/documentation/)



---



