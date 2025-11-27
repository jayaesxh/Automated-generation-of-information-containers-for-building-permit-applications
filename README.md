# Automated Generation of ICDD Containers for Building Permit Applications

This repository contains the code for an end-to-end pipeline that:

1.  Ingests a building-permit submission (ZIP or folder) for a single case.
2.  Builds an **ISO 21597-1 compliant ICDD container** (index.rdf, payload documents, linkset, ontology resources).
3.  Preprocesses all documents into text chunks (JSONL + manifest) for retrieval.
4.  Uses a **Hybrid RAG Strategy** (Semantic Search + Graph RAG) to retrieve context.
5.  Uses a Large Language Model (LLM) to extract key application and building fields.
6.  Converts the extracted fields into an **OntoBPR knowledge graph fragment** (OntoBPR.ttl).
7.  Packs everything into a single `.icdd` container and validates the structure with **SHACL**.

The result is an ICDD container that not only bundles the original permit documents but also contains an RDF graph conformant to OntoBPR, ready for use in downstream reasoning or graph queries.

## 1. Repository structure

```text
icdd-rag-pipeline2/
├─ driver_upload.py          # MAIN ENTRY: runs whole pipeline for all submissions in ./upload
├─ rag_engine.py             # HYBRID RAG: Manages SBERT (Semantic) and LightRAG (Graph) retrieval
├─ ontobpr_llm.py            # EXTRACTION: LLM-based schema extraction + mapping to OntoBPR
├─ field_schema.py           # SCHEMA: Field definitions (application + building)
├─ llm_backend.py            # BACKEND: Wraps the Hugging Face LLM client
├─ requirements.txt          # Dependencies
├─ README.md                 # Documentation
├─ .gitignore
├─ static_resources/
│   └─ ontology_resources/   # ISO 21597-1 Ontologies & SHACL Shapes
│       ├─ Container.rdf
│       ├─ Linkset.rdf
│       ├─ ExtendedLinkset.rdf
│       ├─ Container.shapes.ttl
│       ├─ Part1ClassesCheck.shapes.rdf
│       └─ ...
├─ upload/
│   └─ .gitkeep              # Place your case ZIPs / folders here
└─ output/
    └─ ...                   # Generated cases + ICDD containers
```

## 2. Installation

1.  Clone the repository:
    ```bash
    git clone <YOUR_REPO_URL>.git
    cd icdd-rag-pipeline2
    ```

2.  Create a virtual environment and install dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # on Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **System Requirements:**
    *   Python 3.10+
    *   CUDA-capable GPU recommended for local LLM execution.

## 3. Configuration

The pipeline uses a local or API-based LLM (via Hugging Face) to extract structured fields.

1.  **Environment Variables:**
    Create a `.env` file or export variables:
    ```bash
    export HF_TOKEN="hf_xxx_your_token_here"  # Required for gated models
    ```

2.  **Model Selection:**
    The default model is configured in `llm_backend.py` (e.g., `Qwen/Qwen2.5-7B-Instruct`). You can modify this to use other Hugging Face models.

## 4. Running the pipeline

1.  **Prepare Input:**
    Place a single-case submission (ZIP or folder) into the `upload/` directory:
    ```bash
    cp Apl_25-04543-FULL.zip upload/
    ```

2.  **Run Pipeline:**
    ```bash
    python driver_upload.py
    ```

3.  **Output:**
    For each item in `upload/`, the script generates:
    *   `output/ICDD_<CASE_ID>.icdd`: The final **ISO 21597-1 compliant container**.
    *   `output/<CASE_ID>/`: Unzipped working directory containing:
        *   `index.rdf`: The container index (with `owl:Ontology` header).
        *   `Payload triples/Doc_Application_Links.rdf`: The Linkset (using `ls:hasDocument`).
        *   `Payload triples/OntoBPR.ttl`: The extracted knowledge graph.

## 5. Pipeline Overview

### 5.1 Hybrid Retrieval (RAG)
The system employs a dual-strategy approach via `rag_engine.py`:
*   **Semantic Search (Primary):** Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) to create a dense vector index of document chunks. This ensures robust, high-recall retrieval even on modest hardware.
*   **Experimental Strategy (Graph RAG):** The architecture includes a fully integrated module for Graph-based Retrieval using `LightRAG`. While currently configured as a secondary fallback due to local compute constraints, this demonstrates the system's capability to support advanced graph reasoning.
*   **Fallback Mechanism:** The system prioritizes Semantic Search for stability but can seamlessly fall back or combine results from Graph RAG.

### 5.2 Semantic Extraction (OntoBPR)
*   **Schema:** `field_schema.py` defines the target data structure (Application ID, Site Address, etc.).
*   **Extraction:** `ontobpr_llm.py` uses the retrieved context to prompt the LLM, enforcing a strict JSON output format.
*   **Mapping:** The JSON is mapped to **OntoBPR** classes (`ontobpr:BuildingApplication`, `ontobpr:Building`) and linked to the ICDD container.

### 5.3 ISO 21597-1 Compliance
The generated containers are rigorously compliant:
*   **Index:** Includes `ct:ContainerDescription` with `owl:Ontology` declaration and relative paths.
*   **Linkset:** Implements the official Linkset ontology with `ls:hasDocument` and binary links.
*   **Validation:** The pipeline automatically runs **SHACL validation** (`pyshacl`) to ensure conformance.

## 6. Validation Results
The pipeline has been verified against real-world datasets:
*   **SHACL Conformance:** `True` (Fully Compliant)
*   **Structural Check:** `OK`
*   **Coherence Check:** `OK`

## 7. Future Work
*   **Expanded Ontology:** Extend OntoBPR mapping to cover more complex regulations.
*   **Graph Database:** Integrate Neo4j export for advanced querying of the generated graphs.
*   **Advanced OCR:** Integrate specialized OCR for handwritten engineering drawings.
