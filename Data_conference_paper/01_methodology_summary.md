# Automated Generation of Semantic Information Containers for Building Permit Reviews: A Comprehensive Methodological Framework

## 1. Introduction

### 1.1 Context and Motivation
The architecture, engineering, and construction (AEC) industry faces significant interoperability barriers in regulatory processes. Building permit reviews often involve manual workflows with heterogeneous artifacts (PDFs, IFC models, spreadsheets), leading to delays and opaque decision-making. This research presents an automated pipeline to transform raw submissions into ISO 21597-1 compliant Information Containers for Linked Document Delivery (ICDD), leveraging a Hybrid Retrieval-Augmented Generation (RAG) architecture and Large Language Model (LLM) based semantic extraction.

### 1.2 Objectives
- **Standardization:** Strict adherence to ISO 21597-1 (Container) and ISO 21597-2 (Linkset) standards.
- **Automation:** Minimize human intervention in processing permit applications.
- **Semantic Enrichment:** Extract unstructured data and map it to the OntoBPR ontology.
- **Resilience:** Handle real-world submissions (scanned docs, inconsistent naming).

## 2. Theoretical Framework

### 2.1 The ISO 21597 Information Container (ICDD)
- **Part 1 (Container):** The `index.rdf` acts as the "brain," defining the `ct:ContainerDescription` and aggregating documents (`ct:InternalDocument`).
- **Part 2 (Linkset):** `ls:Link` entities formally assert relationships between documents (e.g., connecting a report to a plan), enabling traceability.

### 2.2 Ontology-Based Knowledge Representation (OntoBPR)
The pipeline normalizes extracted data into the OntoBPR ontology (e.g., `ontobpr:numberOfStoreys` as `xsd:integer`), bridging natural language ambiguity and regulatory logic precision.

### 2.3 Hybrid Retrieval-Augmented Generation (RAG)
To mitigate LLM hallucinations, the system employs a Hybrid RAG approach:
- **Vector Retrieval (Semantic):** Uses dense embeddings (SBERT) to capture semantic similarity.
- **Graph Retrieval (Structural):** Uses a graph database (Neo4j) to capture structural relationships (Document -> Chunk), preserving context often lost in vector-only searches.

## 3. High-Level Architecture and Workflow

The pipeline is orchestrated by a central driver (`driver_upload.py`) and follows a sequential flow:

1.  **Ingestion:** Decompression, sanitization, and `index.rdf` generation.
2.  **Preprocessing:** Text mining (PDFMiner/OCR) and chunking.
3.  **Hybrid Retrieval:** Parallel Vector (SBERT) and Graph (Neo4j) indexing.
4.  **Semantic Extraction:** Schema-guided LLM (Qwen 2.5) extraction.
5.  **Ontology Mapping:** JSON to RDF (OntoBPR) transformation and Linkset generation.
6.  **Packaging:** Serialization into `.icdd` container with SHACL validation.

## 4. Methodology Components

### 4.1 Component I: Ingestion and Containerization
- **Staging:** `driver_upload.py` implements a "quarantine and sanitize" protocol, filtering system artifacts (e.g., `__pycache__`) and normalizing filenames.
- **RDF Generation:** Dynamically constructs `index.rdf`. Heuristics (`_guess_mime`) assign `ct:format` and `ct:filetype`. The container is explicitly typed with `ct:conformanceIndicator` "ICDD-Part1-Container".

### 4.2 Component II: Multi-Modal Preprocessing
- **PDF Processing:** Implements a three-tier fallback:
    1.  **PDFMiner:** Direct text extraction (high fidelity).
    2.  **OCR (Tesseract):** Image-based extraction for scanned documents.
    3.  **Preview:** Raw byte preview for resilience against corrupted files.
- **Chunking:** Role-based chunking assigns metadata (e.g., "Building Application", "Site Plan") based on filename patterns, aiding retrieval weighting.

### 4.3 Component III: The Hybrid RAG Engine
- **Vector Retrieval:** Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) for cosine similarity-based semantic search.
- **Graph Retrieval:** `neo4j_push.py` populates a Neo4j graph (`(:BuildingApplication) --> (:Document) --> (:Chunk)`). `duel_retriever.py` fuses context from both streams, providing the LLM with semantically similar *and* structurally relevant text.

### 4.4 Component IV: Schema-Guided Semantic Extraction
- **Inference Engine:** Qwen 2.5-7B-Instruct is used for its strong instruction-following and JSON generation capabilities.
- **Strict Schema:** `field_schema.py` defines the target data structure (e.g., `gross_floor_area_m2` as Number).
- **Prompt Engineering:** Uses a "Template Filling" strategy. The prompt enforces "Use ONLY the information from the following context" and requires valid JSON output, minimizing hallucinations.

### 4.5 Component V: Ontology Mapping
- **Mapping:** `schema_to_ontobpr` transforms the LLM's JSON output into RDF triples (e.g., JSON `site_address` $\rightarrow$ `ontobpr:siteAddress`).
- **Linksets:** `build_payload_linkset` generates `Doc_Application_Links.rdf` (ISO 21597-2), creating `DocDocLink` relationships between documents for traceability.

### 4.6 Component VI: Validation and SHACL Conformance
- **SHACL Validation:** Uses `pyshacl` to validate against ISO 21597 shapes.
- **Dynamic Relaxation:** The system programmatically relaxes "Closed World" constraints (`sh:closed`, `sh:ignoredProperties`) in memory during validation. This prevents failures due to benign metadata additions (e.g., provenance info) while ensuring structural completeness.

## 5. Results and Conclusion

### 5.1 Results
- **Extraction:** The pipeline successfully normalizes free text into structured OntoBPR properties (e.g., parsing "3 George Street..." into `ontobpr:siteAddress`).
- **Conformance:** Generated containers pass ISO 21597-1 SHACL validation (after dynamic relaxation), confirming correct class typing and structural relationships.

### 5.2 Contributions
- **End-to-End Automation:** Fully automated creation of ISO 21597 containers.
- **Hybrid RAG:** Combines vector and graph retrieval for robust context.
- **Schema-Guided Extraction:** Ensures interoperable, ontology-aligned output.

### 5.3 Conclusion
The methodology successfully bridges the gap between unstructured submissions and structured regulatory data by synergizing ISO standards with LLM capabilities. Future work includes automated compliance checking using the generated OntoBPR graph and integrating Vision-Language models for drawing interpretation.
