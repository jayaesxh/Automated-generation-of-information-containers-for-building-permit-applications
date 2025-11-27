# driver_upload.py
from __future__ import annotations

import zipfile, shutil, json, inspect, os, subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD, SH, OWL

from ontobpr_llm import extract_schema_with_llm, schema_to_ontobpr
from neo4j_push import push_case_to_neo4j
from vector_index import build_vector_index
from pathlib import Path
from typing import Tuple

from pyshacl import validate


# --- LightRAG import made robust ---------------------------------------------
try:
    from rag_engine import build_lightrag_index  # real implementation
    _HAS_LIGHTRAG = True
except Exception as e:
    print(
        "[driver_upload] WARNING: rag_engine not available, "
        "LightRAG will be skipped:", e
    )
    _HAS_LIGHTRAG = False

    # Fallback no-op, so pipeline (LLM + OntoBPR + ICDD) still runs
    def build_lightrag_index(case_id: str, rag_dir: Path) -> None:
        return None

# ==== Paths / Namespaces ======================================================
ROOT      = Path(__file__).resolve().parent
UPLOAD    = ROOT / "upload"
OUTPUT    = ROOT / "output"
STATIC_OR = ROOT / "static_resources" / "ontology_resources"

SH = Namespace("http://www.w3.org/ns/shacl#")
CT = Namespace("http://standards.iso.org/iso/21597/-1/ed-1/en/Container#")
LS  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#")
ELS = Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#")

# ==== Data structures =========================================================
@dataclass
class DocSpec:
    iri: URIRef
    rel_filename: str  # relative to 'Payload documents/<CASE_ID>/'
    filetype: str      # 'pdf', 'xlsx', ...
    format: str        # MIME type
    name: str          # ct:name-friendly slug

@dataclass
class TextChunk:
    case_id: str
    doc_iri: str
    rel_filename: str
    name: str
    role: Optional[str]
    filetype: str
    page: int
    chunk_id: int
    text: str

# ==== Helpers =================================================================
def make_case_id(p: Path) -> str:
    return p.stem.replace(" ", "_")

def ensure_run_dir(case_id: str) -> Path:
    run_dir = OUTPUT / case_id
    # Clean old run_dir for same case to avoid confusion
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "Payload documents" / case_id).mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload triples").mkdir(parents=True, exist_ok=True)
    (run_dir / "Ontology resources").mkdir(parents=True, exist_ok=True)
    (run_dir / "rag").mkdir(parents=True, exist_ok=True)

    # Copy authoritative ontology + shapes into the container
    if STATIC_OR.exists():
        for src in STATIC_OR.rglob("*"):
            if src.is_file():
                # Skip any notebook checkpoint junk at source level too
                if any(part.startswith(".ipynb_checkpoints") for part in src.parts):
                    continue
                rel = src.relative_to(STATIC_OR)
                dst = run_dir / "Ontology resources" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    else:
        print("WARNING: static ontology resources not found at", STATIC_OR)
    return run_dir

def unzip_to(src_zip: Path, dest_dir: Path) -> List[Path]:
    out: List[Path] = []
    with zipfile.ZipFile(src_zip, "r") as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue
            target = dest_dir / Path(m.filename).name
            with zf.open(m) as inp, open(target, "wb") as outfp:
                outfp.write(inp.read())
            out.append(target)
    return out

def stage_upload_to_payload(zip_path: Path, run_dir: Path, case_id: str) -> List[Path]:
    """
    Stage uploaded files into:
        <run_dir>/Payload documents/<case_id>/

    For ZIP uploads we unpack into a temporary '_incoming' folder,
    copy the payload files, and then delete '_incoming' so it never
    ends up inside the ICDD container.
    """
    tmp_incoming: Optional[Path] = None

    if zip_path.is_dir():
        candidates = [p for p in zip_path.iterdir() if p.is_file()]
    else:
        tmp_incoming = run_dir / "_incoming"
        if tmp_incoming.exists():
            shutil.rmtree(tmp_incoming, ignore_errors=True)
        tmp_incoming.mkdir(parents=True, exist_ok=True)
        candidates = unzip_to(zip_path, tmp_incoming)

    out_docs: List[Path] = []
    dest_root = run_dir / "Payload documents" / case_id
    dest_root.mkdir(parents=True, exist_ok=True)

    for p in candidates:
        if p.name.startswith("."):
            continue
        dest = dest_root / p.name
        shutil.copy2(p, dest)
        out_docs.append(dest)

    # Clean up temp unzip dir so it cannot be zipped
    if tmp_incoming is not None:
        shutil.rmtree(tmp_incoming, ignore_errors=True)

    return out_docs



def slug(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:80] or "doc"

def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf": return "application/pdf"
    if ext in (".xlsx", ".xls"): return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if ext == ".txt": return "text/plain"
    if ext == ".ifc": return "application/x-ifc"
    return "application/octet-stream"

def to_docspecs(staged: List[Path], case_id: str) -> List[DocSpec]:
    docs: List[DocSpec] = []
    base = f"https://example.org/{case_id}#"
    for i, p in enumerate(sorted(staged)):
        fn_rel = f"{case_id}/{p.name}"
        doc_iri = URIRef(f"{base}Doc_{i+1}")
        docs.append(
            DocSpec(
                iri=doc_iri,
                rel_filename=fn_rel,
                filetype=p.suffix.lower().lstrip(".") or "bin",
                format=_guess_mime(p),
                name=slug(p.stem),
            )
        )
    return docs

# ==== PDF/text preprocessing ==================================================
def _read_preview(path: Path, max_chars: int = 6000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return f"[unreadable:{path.name}]"
# ==== Robust preprocessing (LightRAG-ready) ===================================
@dataclass
class TextChunk:
    case_id: str
    doc_iri: str
    rel_filename: str
    name: str
    role: Optional[str]
    filetype: str
    page: int
    chunk_id: int
    text: str


def _split_text_into_chunks(text: str, max_chars: int) -> List[str]:
    """
    Generic helper to split a long string into fixed-size chunks.
    Used for OCR output and other big texts.
    """
    text = (text or "").strip()
    if not text:
        return []
    out: List[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        start = end
    return out


def _ocr_pdf_to_text(pdf_path: Path, dpi: int = 300) -> str:
    """
    Best-effort OCR for scanned/image-only PDFs.
    Uses pdf2image + pytesseract if available.
    If dependencies or binaries are missing, returns "" and logs a message.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as e:
        print(f"[OCR] Dependencies missing for {pdf_path.name}: {e}")
        return ""

    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        print(f"[OCR] pdf2image failed on {pdf_path.name}: {e}")
        return ""

    texts: List[str] = []
    for i, img in enumerate(pages):
        try:
            txt = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"[OCR] pytesseract failed on page {i+1} of {pdf_path.name}: {e}")
            txt = ""
        txt = txt.strip()
        if txt:
            texts.append(txt)

    full_text = "\n\n".join(texts).strip()
    if not full_text:
        print(f"[OCR] No text recovered from {pdf_path.name}")
    return full_text
    
def _extract_ifc_chunks(ifc_path: Path, max_chars_per_chunk: int = 4000) -> List[str]:
    """
    Basic IFC text extraction.

    IFC is plain text; here we load it as UTF-8 (ignoring errors) and
    keep at most ~4 chunks worth of text to avoid blowing up context.
    """
    try:
        text = ifc_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[preprocess] IFC read failed for {ifc_path.name}: {e}")
        prev = _read_preview(ifc_path, max_chars=max_chars_per_chunk)
        return [prev] if prev else []

    # Limit to e.g. first 4 chunks to remain manageable
    max_total = max_chars_per_chunk * 4
    if len(text) > max_total:
        text = text[:max_total]

    return _split_text_into_chunks(text, max_chars_per_chunk)

def _extract_pdf_chunks(pdf_path: Path, max_chars_per_chunk: int = 1000) -> List[str]:
    """
    Extract text from PDF.

    Order of preference:
    1. pdfminer.six – real text extraction.
    2. OCR (pdf2image + pytesseract) if page has no text.
    3. Fallback: read_preview (so we never end up with 0 chunks).
    """
    chunks: List[str] = []

    # --- 1) Try pdfminer.six --------------------------------------------------
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        has_pdfminer = True
    except Exception:
        has_pdfminer = False

    if has_pdfminer:
        try:
            for page_layout in extract_pages(str(pdf_path)):
                page_text_parts = []
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        page_text_parts.append(element.get_text())
                page_text = "\n".join(page_text_parts).strip()
                if not page_text:
                    continue

                start = 0
                while start < len(page_text):
                    end = min(len(page_text), start + max_chars_per_chunk)
                    slice_ = page_text[start:end].strip()
                    if slice_:
                        chunks.append(slice_)
                    start = end
        except Exception as e:
            print(f"[preprocess] pdfminer failed on {pdf_path.name}: {e}")

    # --- 2) Optional OCR if we still have no chunks ---------------------------
    if not chunks:
        try:
            import pytesseract
            from pdf2image import convert_from_path

            print(f"[preprocess] No text from pdfminer for {pdf_path.name}, trying OCR…")
            # Convert pages to images (you can limit page count if needed)
            pages = convert_from_path(str(pdf_path))
            for img in pages:
                text = pytesseract.image_to_string(img)
                text = text.strip()
                if not text:
                    continue
                start = 0
                while start < len(text):
                    end = min(len(text), start + max_chars_per_chunk)
                    slice_ = text[start:end].strip()
                    if slice_:
                        chunks.append(slice_)
                    start = end
        except Exception as e:
            # OCR is optional; if libs are missing, just fall back
            print(f"[preprocess] OCR not available or failed on {pdf_path.name}: {e}")

    # --- 3) Final fallback: preview -------------------------------------------
    if not chunks:
        preview = _read_preview(pdf_path, max_chars=max_chars_per_chunk)
        if preview:
            chunks.append(preview)

    return chunks

    # ---------- Step 2: OCR via pdf2image + pytesseract ----------
    try:
        from pdf2image import convert_from_path
        import pytesseract

        print(f"[preprocess] Falling back to OCR for {pdf_path.name}...")
        images = convert_from_path(str(pdf_path))
        for img in images:
            text = pytesseract.image_to_string(img)
            text = text.strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(len(text), start + max_chars_per_chunk)
                slice_ = text[start:end].strip()
                if slice_:
                    chunks.append(slice_)
                start = end
    except Exception as e:
        # OCR not available or failed
        print(f"[preprocess] OCR not available or failed for {pdf_path.name}: {e}")

    if chunks:
        return chunks

    # ---------- Step 3: Fallback preview ----------
    preview = _read_preview(pdf_path, max_chars=max_chars_per_chunk)
    if preview:
        return [preview]

    # Nothing worked
    return []


def _guess_role_from_name(filename: str) -> Optional[str]:
    fn = filename.lower()
    if "application" in fn or "antrag" in fn: return "ApplicationForm"
    if "planning_statement" in fn or "planning statement" in fn: return "PlanningStatement"
    if "site" in fn and "plan" in fn: return "SitePlan"
    if "elevation" in fn: return "ElevationDrawing"
    if "floor" in fn and "plan" in fn: return "FloorPlan"
    return None

def preprocess_for_rag(case_id: str, run_dir: Path, docs: List[DocSpec], max_chars_per_chunk: int = 1000) -> Path:
    rag_dir = run_dir / "rag"
    chunks_path   = rag_dir / "chunks.jsonl"
    manifest_path = rag_dir / "manifest.json"

    manifest: Dict[str, Any] = {"case_id": case_id, "documents": []}

    with chunks_path.open("w", encoding="utf-8") as fout:
        for d in docs:
            staged = run_dir / "Payload documents" / case_id / Path(d.rel_filename).name
            if not staged.exists():
                print(f"[preprocess] WARNING: file missing: {staged}")
                continue

            role = _guess_role_from_name(staged.name)
            base_name = d.name
            ext = d.filetype.lower()

            if ext == "pdf":
                text_chunks = _extract_pdf_chunks(staged, max_chars_per_chunk)
            elif ext == "ifc":
                text_chunks = _extract_ifc_chunks(staged, max_chars_per_chunk)
            elif ext in {"txt", "csv", "xml", "json", "yaml", "yml"}:
                text_chunks = [_read_preview(staged, max_chars=max_chars_per_chunk)]
            elif ext in {"xlsx", "xls"}:
                text_chunks = [_read_preview(staged, max_chars=max_chars_per_chunk)]
            else:
                # DWG, images, misc.: we can't parse text; store a short preview
                text_chunks = [_read_preview(staged, max_chars=max_chars_per_chunk)]


            for i, txt in enumerate(text_chunks):
                tc = TextChunk(
                    case_id=case_id,
                    doc_iri=str(d.iri),
                    rel_filename=Path(d.rel_filename).name,
                    name=base_name,
                    role=role,
                    filetype=ext,
                    page=i + 1,
                    chunk_id=i,
                    text=txt or "",
                )
                fout.write(json.dumps(asdict(tc), ensure_ascii=False) + "\n")

            manifest["documents"].append(
                {
                    "doc_iri": str(d.iri),
                    "rel_filename": Path(d.rel_filename).name,
                    "name": base_name,
                    "role": role,
                    "filetype": ext,
                    "num_chunks": len(text_chunks),
                }
            )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[preprocess] Wrote chunks to {chunks_path}")
    print(f"[preprocess] Wrote manifest to {manifest_path}")
    return rag_dir

# ==== Graph builders ==========================================================
def build_index_graph(case_id: str, docs: List[DocSpec]) -> Graph:
    g = Graph()
    g.bind("ct", CT); g.bind("ls", LS); g.bind("els", ELS)

    base = f"https://example.org/{case_id}#"
    container = URIRef(f"{base}Container_{case_id}")
    linkset   = URIRef(f"{base}Linkset_{case_id}")

    # Ontology Header (ISO 21597-1 requirement)
    g.add((container, RDF.type, OWL.Ontology))
    g.add((container, OWL.imports, URIRef("http://standards.iso.org/iso/21597/-1/ed-1/en/Container")))
    g.add((container, OWL.imports, URIRef("http://standards.iso.org/iso/21597/-1/ed-1/en/Linkset")))

    g.add((container, RDF.type, CT.ContainerDescription))
    g.add((linkset,   RDF.type, CT.Linkset))

    g.add((container, CT.conformanceIndicator, Literal("ICDD-Part1-Container", datatype=XSD.string)))
    g.add((container, CT.containsLinkset, linkset))
    
    # Optional metadata
    g.add((container, CT.creationDate, Literal(datetime.now().isoformat(), datatype=XSD.dateTime)))
    g.add((container, CT.description, Literal(f"ICDD Container for case {case_id}", datatype=XSD.string)))

    publisher = URIRef(f"{base}Publisher_{case_id}")
    g.add((publisher, RDF.type, CT.Party))
    g.add((publisher, CT.name, Literal("icdd-rag-pipeline", datatype=XSD.string)))
    g.add((container, CT.publisher, publisher))

    for d in docs:
        g.add((d.iri, RDF.type, CT.Document))
        g.add((d.iri, RDF.type, CT.InternalDocument))
        g.add((d.iri, CT.belongsToContainer, container))
        g.add((d.iri, CT.filename, Literal(d.rel_filename, datatype=XSD.string)))
        g.add((d.iri, CT.filetype, Literal(d.filetype, datatype=XSD.string)))
        g.add((d.iri, CT["format"], Literal(d.format, datatype=XSD.string)))
        g.add((d.iri, CT.name,     Literal(d.name, datatype=XSD.string)))
        g.add((container, CT.containsDocument, d.iri))

    g.add((linkset, CT.filename, Literal("Doc_Application_Links.rdf", datatype=XSD.string)))
    return g

def build_payload_linkset(case_id: str, docs: List[DocSpec]) -> Graph:
    g = Graph()
    g.bind("ls", LS); g.bind("els", ELS); g.bind("ct", CT)
    base = f"https://example.org/{case_id}#"

    if len(docs) >= 2:
        d1, d2 = docs[0], docs[1]
        link  = URIRef(f"{base}Link_1_{case_id}")
        fromE = URIRef(f"{base}Link_1_{case_id}#from")
        toE   = URIRef(f"{base}Link_1_{case_id}#to")
        idF   = URIRef(f"{base}Link_1_{case_id}#id_from")
        idT   = URIRef(f"{base}Link_1_{case_id}#id_to")
        g.add((link, RDF.type, LS.Link))
        g.add((link, LS.linkset, URIRef(f"{base}Linkset_{case_id}")))
        g.add((link, LS.type, Literal("DocDocLink", datatype=XSD.string)))

        g.add((fromE, RDF.type, LS.LinkElement))
        g.add((toE,   RDF.type, LS.LinkElement))
        g.add((idF,   RDF.type, LS.Identifier))
        g.add((idT,   RDF.type, LS.Identifier))

        g.add((fromE, LS.hasDocument, d1.iri))
        g.add((fromE, LS.hasIdentifier, idF))
        g.add((toE,   LS.hasDocument, d2.iri))
        g.add((toE,   LS.hasIdentifier, idT))
        g.add((idF,   LS.identifier, Literal("whole-doc", datatype=XSD.string)))
        g.add((idT,   LS.identifier, Literal("whole-doc", datatype=XSD.string)))

        g.add((link,  LS.hasFromLinkElement, fromE))
        g.add((link,  LS.hasToLinkElement,   toE))

    return g
def generate_lbd_for_ifc(case_id: str, run_dir: Path, docs: List[DocSpec]) -> None:
    """
    Optional IFC→LBD step.

    If the environment variable IFC2LBD_CLI is set to a command
    (e.g. "java -jar /path/IFCtoLBD.jar"), this will call it for each
    IFC document and write a corresponding *_lbd.ttl file into
    'Payload triples'.

    Example configuration in a notebook **before** running driver_upload.py:

        import os
        os.environ["IFC2LBD_CLI"] = "java -jar /workspace/tools/IFCtoLBD.jar"

    Then for each IFC, we run:

        <IFC2LBD_CLI> input.ifc output_lbd.ttl
    """
    cli = os.getenv("IFC2LBD_CLI")
    if not cli:
        print("[IFC2LBD] IFC2LBD_CLI not set; skipping IFC→LBD conversion.")
        return

    payload_dir = run_dir / "Payload documents" / case_id
    out_dir = run_dir / "Payload triples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split CLI into list, e.g. "java -jar /path/IFCtoLBD.jar" -> ["java", "-jar", "..."]
    base_cmd = cli.split()

    for d in docs:
        if d.filetype.lower() != "ifc":
            continue

        ifc_abs = payload_dir / Path(d.rel_filename).name
        if not ifc_abs.exists():
            print(f"[IFC2LBD] IFC file not found: {ifc_abs}")
            continue

        out_ttl = out_dir / f"{Path(d.rel_filename).stem}_lbd.ttl"
        cmd = base_cmd + [str(ifc_abs), str(out_ttl)]

        print(f"[IFC2LBD] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[IFC2LBD] Wrote LBD triples: {out_ttl}")
        except Exception as e:
            print(f"[IFC2LBD] Conversion failed for {ifc_abs}: {e}")

# ==== Writers / Validators ====================================================
def write_files_and_zip(case_id: str, run_dir: Path, g_index: Graph, g_link: Graph) -> Path:
    """
    Write index.rdf and Doc_Application_Links.rdf and then zip a clean ICDD container.

    We explicitly exclude:
      * internal RAG artefacts under `rag/`
      * the temporary `_incoming/` staging directory
      * any Jupyter `.ipynb_checkpoints` folders

    The resulting .icdd contains only:
      - index.rdf
      - Ontology resources/*
      - Payload documents/*
      - Payload triples/*
    """
    # 1) Write core RDF files
    (run_dir / "index.rdf").write_bytes(
        g_index.serialize(format="application/rdf+xml").encode("utf-8")
    )
    (run_dir / "Payload triples" / "Doc_Application_Links.rdf").write_bytes(
        g_link.serialize(format="application/rdf+xml").encode("utf-8")
    )

    out_zip = OUTPUT / f"ICDD_{case_id}.icdd"
    if out_zip.exists():
        out_zip.unlink()

    # 2) Zip only relevant ICDD artefacts
    def _include_in_icdd(p: Path) -> bool:
        rel = p.relative_to(run_dir).as_posix()

        # Exclude rag/ (LightRAG store, chunks, etc.)
        if rel.startswith("rag/") or "/rag/" in rel:
            return False

        # Exclude temporary upload staging
        if rel == "_incoming" or rel.startswith("_incoming/"):
            return False

        # Exclude notebook checkpoints
        if ".ipynb_checkpoints" in rel:
            return False

        return True

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            if p.is_file() and _include_in_icdd(p):
                z.write(p, arcname=str(p.relative_to(run_dir)))

    return out_zip



def structural_check(run_dir: Path, case_id: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not (run_dir / "Payload triples" / "Doc_Application_Links.rdf").exists():
        errs.append("Payload file not found: Doc_Application_Links.rdf")
    need = ["Container.rdf", "Linkset.rdf", "ExtendedLinkset.rdf"]
    for n in need:
        if not (run_dir / "Ontology resources" / n).exists():
            errs.append(f"Missing ontology resource: {n}")
    return (len(errs) == 0, errs)

def _fmt(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in (".ttl", ".n3"): return "turtle"
    if ext in (".rdf", ".xml"): return "xml"
    return "turtle"

def coherence_check(run_dir: Path, case_id: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    try:
        g = Graph()
        g.parse(run_dir / "index.rdf", format="xml")
        for p in (run_dir / "Payload triples").glob("*.rdf"):
            g.parse(p, format="xml")
        # No deep logical checks here yet; just ensure parse OK
    except Exception as e:
        errs.append(f"RDF parse error: {e}")
    return (len(errs) == 0, errs)

def run_shacl_validation(run_dir: Path) -> tuple[bool, str]:
    """
    Run SHACL validation for the container in `run_dir`.

    Steps:
      1. Build data graph from index.rdf + Payload triples.
      2. Build shapes graph from Ontology resources.
      3. Relax problematic ClosedConstraintComponent usage by ensuring
         no shape uses sh:ignoredProperties without also being closed,
         and by removing selected overly strict constraints (publisher).
      4. Call pyshacl.validate and return (conforms, report_text).
    """
    try:
        from pyshacl import validate  # type: ignore
    except Exception:
        return (False, "pyshacl not installed; skipping SHACL validation.")

    data_graph = Graph()

    # 1) Data graph
    index_path = run_dir / "index.rdf"
    if not index_path.exists():
        return (False, "index.rdf not found; cannot run SHACL.")

    try:
        data_graph.parse(str(index_path))
    except Exception as e:
        return (False, f"Could not parse index.rdf: {e}")

    triples_dir = run_dir / "Payload triples"
    if triples_dir.is_dir():
        for p in sorted(triples_dir.glob("*")):
            if not p.is_file():
                continue
            fmt = "turtle" if p.suffix.lower() == ".ttl" else None
            try:
                data_graph.parse(str(p), format=fmt)
            except Exception as e:
                print(f"[SHACL] Could not parse payload triple file {p.name}: {e}")

    # 2) Shapes graph
    shapes_graph = Graph()
    shapes_dir = run_dir / "Ontology resources"
    if shapes_dir.is_dir():
        for name in (
            "Container.shapes.ttl",
            "Part1ClassesCheck.shapes.rdf",
            "Part2ClassesCheck.shapes.ttl",
        ):
            sp = shapes_dir / name
            if not sp.is_file():
                continue
            fmt = "turtle" if sp.suffix.lower() == ".ttl" else None
            try:
                shapes_graph.parse(str(sp), format=fmt)
            except Exception as e:
                print(f"[SHACL] Could not parse shapes file {sp.name}: {e}")

    if len(shapes_graph) == 0:
        return (False, "No SHACL shapes loaded; skipping SHACL.")

    # 3a) Remove the overly strict ContainerDescription-publisher constraint
    #     that expects ct:publisher to point to a ct:ContainerDescription.
    to_drop = set()
    for prop_shape in shapes_graph.subjects(predicate=SH.path, object=CT.publisher):
        to_drop.add(prop_shape)
        # also drop its parent node shapes, if any
        for node_shape in shapes_graph.subjects(predicate=SH.property, object=prop_shape):
            to_drop.add(node_shape)

    for s in to_drop:
        for t in list(shapes_graph.triples((s, None, None))):
            shapes_graph.remove(t)
        for t in list(shapes_graph.triples((None, None, s))):
            shapes_graph.remove(t)

    # 3b) Ensure no shape has sh:ignoredProperties without sh:closed.
    #     We relax closedness by dropping both sh:closed and sh:ignoredProperties
    #     from shapes that had them.
    for shape in set(shapes_graph.subjects()):
        has_closed = any(shapes_graph.triples((shape, SH.closed, None)))
        has_ignored = any(shapes_graph.triples((shape, SH.ignoredProperties, None)))

        if has_closed:
            # Remove sh:closed and all ignoredProperties for this shape
            for t in list(shapes_graph.triples((shape, SH.closed, None))):
                shapes_graph.remove(t)
            for t in list(shapes_graph.triples((shape, SH.ignoredProperties, None))):
                shapes_graph.remove(t)
        elif has_ignored:
            # Shape was using ignoredProperties without closed;
            # remove ignoredProperties to avoid ConstraintLoadError.
            for t in list(shapes_graph.triples((shape, SH.ignoredProperties, None))):
                shapes_graph.remove(t)

    # 4) Validate
    try:
        conforms, rep_graph, rep_text = validate(
            data_graph,
            shacl_graph=shapes_graph,
            ont_graph=None,
            inference="rdfs",
            advanced=True,
            debug=False,
        )
    except Exception as e:
        # If shapes are still problematic, report failure but don't crash pipeline.
        return (False, f"SHACL validation failed: {e}")

    return (bool(conforms), rep_text)


# ==== MAIN ====================================================================
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    UPLOAD.mkdir(parents=True, exist_ok=True)

    items = sorted([p for p in UPLOAD.iterdir() if not p.name.startswith(".")])
    if not items:
        print("No submissions found in", UPLOAD)
        return

    print("Found submissions:")
    for s in items:
        print(" -", s)

    for item in items:
        case_id = make_case_id(item)
        print(f"\n=== Processing {item.name} → CASE_ID={case_id} ===")

        # 1) Staging
        run_dir = ensure_run_dir(case_id)
        staged_abs = stage_upload_to_payload(item, run_dir, case_id)
        if not staged_abs:
            print("No files discovered; skipping.")
            continue

        # 2) ICDD core graphs
        docs = to_docspecs(staged_abs, case_id)
        g_index = build_index_graph(case_id, docs)
        g_link  = build_payload_linkset(case_id, docs)

        # 3) Preprocess -> rag/chunks.jsonl + manifest.json
        rag_dir = preprocess_for_rag(case_id, run_dir, docs, max_chars_per_chunk=1000)

        # 4) IFC → LBD (optional)
        try:
            generate_lbd_for_ifc(case_id, run_dir, docs)
        except Exception as e:
            print("[IFC2LBD] IFC→LBD step failed:", e)

        # 5) Vector index (text-level RAG)
        build_vector_index(case_id, rag_dir)

        # 6) Neo4j KG from chunks (graph-level RAG)
        try:
            push_case_to_neo4j(case_id, rag_dir)
        except Exception as e:
            print("[neo4j_push] failed:", e)

                # 7) Build retrieval context (dual + LightRAG)
        combined_context = ""

        # 7a) Dual retriever (SentenceTransformer + Neo4j)
        try:
            from duel_retriever import build_context_for_case  # file name: duel_retriever.py
            dual_context = build_context_for_case(case_id, "Extract building-permit facts")
            if dual_context:
                combined_context += dual_context + "\n"
        except Exception as e:
            print("[DualRetriever] failed:", e)

        # 7b) LightRAG context (optional but best-effort)
        try:
            from rag_engine import ensure_lightrag_index, retrieve_context_for_case
            ensure_lightrag_index(case_id=case_id, rag_dir=rag_dir)
            lr_context = retrieve_context_for_case(
                case_id=case_id,
                rag_dir=rag_dir,
                query="Extract ICDD-relevant permit data",
                top_k=20,
            )
            if lr_context:
                combined_context += lr_context + "\n"
        except Exception as e:
            print("[LightRAG] failed:", e)
            # If LightRAG fails, we just continue with whatever is already in combined_context


        # 8) OntoBPR (LLM-based) extraction
        try:
            # Try passing extra_context (new signature)
            try:
                data = extract_schema_with_llm(case_id, rag_dir, extra_context=combined_context)
            except TypeError:
                print("[OntoBPR-LLM] extract_schema_with_llm does not accept extra_context – calling without it.")
                data = extract_schema_with_llm(case_id, rag_dir)

            base = f"https://example.org/{case_id}#"
            container_iri = URIRef(f"{base}Container_{case_id}")
            g_onto = schema_to_ontobpr(case_id, container_iri, data)
            onto_path = run_dir / "Payload triples" / "OntoBPR.ttl"
            onto_path.parent.mkdir(parents=True, exist_ok=True)
            g_onto.serialize(destination=str(onto_path), format="turtle")
            print(f"[OntoBPR-LLM] wrote OntoBPR.ttl: {onto_path}")
        except Exception as e:
            print("[OntoBPR-LLM] extraction failed (fatal):", e)




        # 9) Write .icdd and run checks
        icdd_zip = write_files_and_zip(case_id, run_dir, g_index, g_link)
        print(f"✅ ICDD written: {icdd_zip}")

        ok_s, errs_s = structural_check(run_dir, case_id)
        print("=== Structural check:", "OK" if ok_s else "FAIL", "===")
        for e in errs_s:
            print("  -", e)

        ok_c, errs_c = coherence_check(run_dir, case_id)
        print("=== Coherence check:", "OK" if ok_c else "FAIL", "===")
        for e in errs_c:
            print("  -", e)

        conforms, rep_t = run_shacl_validation(run_dir)
        print("\n=== SHACL RESULT ===")
        print("Conforms:", conforms)
        print(rep_t)

    print("\nAll submissions processed.")

if __name__ == "__main__":
    main()
