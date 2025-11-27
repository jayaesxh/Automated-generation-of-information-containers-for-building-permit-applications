# ontobpr_llm.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

from field_schema import default_schema
from llm_backend import call_llm

# Dual retriever (vector + Neo4j)
from duel_retriever import build_context_for_case  # module is 'duel_retriever.py'

# LightRAG (optional)
try:
    from rag_engine import retrieve_context_for_case
    _HAS_LIGHTRAG = True
except Exception:
    retrieve_context_for_case = None
    _HAS_LIGHTRAG = False

ONTOBPR = Namespace("https://w3id.org/ontobpr#")
CT      = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")


def _load_chunks_for_case(rag_dir: Path) -> List[Dict[str, Any]]:
    chunks_path = rag_dir / "chunks.jsonl"
    chunks: List[Dict[str, Any]] = []
    if not chunks_path.exists():
        return []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def extract_schema_with_llm(
    case_id: str,
    rag_dir: Path,
    extra_context: str | None = None,
) -> Dict[str, Any]:
    """
    Main semantic extraction: dual retrieval (vector+KG) + LightRAG + raw chunks.
    Returns a dict matching the JSON schema (application + building).
    """
    schema = default_schema()
    rag_dir = Path(rag_dir)
    chunks = _load_chunks_for_case(rag_dir)

    # ---- 1) Dual retrieval: vector + KG ----
    try:
        dual_ctx = build_context_for_case(case_id, "Extract core building-permit facts")
    except Exception as e:
        print("[DualRetriever] failed:", e)
        dual_ctx = ""

    # ---- 2) LightRAG (optional) ----
    lr_ctx = ""
    if retrieve_context_for_case:
        try:
            lr_ctx = retrieve_context_for_case(
                case_id=case_id,
                rag_dir=rag_dir,
                query="Extract all information relevant to building application",
                top_k=20,
            )
        except Exception as e:
            print("[LightRAG] retrieval failed:", e)
            lr_ctx = ""

    # ---- 3) Sample some raw chunks ----
    raw_sample = " ".join(c.get("text", "") for c in chunks[:5])

    # ---- 4) Combine all contexts ----
    full_context = f"""
=== EXTRA_CONTEXT (from driver) ===
{extra_context or ""}

=== DUAL_KG_VECTOR_RETRIEVAL ===
{dual_ctx}

=== LIGHTRAG ===
{lr_ctx}

=== RAW CHUNKS (sample) ===
{raw_sample}
"""

    # ---- 5) Build JSON template ----
    template: Dict[str, Any] = {
        "case_id": case_id,
        "building_application": {f.id: None for f in schema.application_fields},
        "building": {f.id: None for f in schema.building_fields},
    }

    field_desc_lines: List[str] = []
    for f in schema.application_fields + schema.building_fields:
        desc = f"- {f.id} ({f.answer_type}"
        if f.enum_values:
            desc += f", one of: {', '.join(f.enum_values)}"
        desc += f") – {f.question}"
        field_desc_lines.append(desc)

    prompt = f"""
Use ONLY the information from the following context to fill the JSON fields.
If a field is not mentioned, set it to null. Do not hallucinate.

FIELDS:
{chr(10).join(field_desc_lines)}

JSON TEMPLATE:
{json.dumps(template, indent=2)}

CONTEXT:
\"\"\" 
{full_context}
\"\"\"
"""

    messages = [
        {"role": "system", "content": "Extract precise structured data. Return ONLY JSON."},
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(messages)

    try:
        raw = raw.strip().replace("```json", "").replace("```", "")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON is not an object.")
        return data
    except Exception as e:
        print("[ontobpr_llm] JSON parsing failed, returning template. Error:", e)
        return template


def schema_to_ontobpr(case_id: str, container_iri: URIRef, data: Dict[str, Any]) -> Graph:
    """
    Map the extracted schema into OntoBPR triples.
    """
    g = Graph()
    g.bind("ontobpr", ONTOBPR)
    g.bind("ct", CT)

    base = f"https://example.org/{case_id}#"
    ba   = URIRef(f"{base}BA_{case_id}")
    bldg = URIRef(f"{base}Building_{case_id}")

    g.add((ba, RDF.type, ONTOBPR.BuildingApplication))
    g.add((ba, ONTOBPR.hasBuildingApplicationContainer, container_iri))

    app = data.get("building_application", {}) or {}
    bld = data.get("building", {}) or {}

    def _lit(subj, pred, value, dt=None):
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        if dt:
            g.add((subj, pred, Literal(value, datatype=dt)))
        else:
            g.add((subj, pred, Literal(value)))

    # Minimal mappings (can be extended)
    _lit(ba, ONTOBPR.applicationIdentifier, app.get("application_reference"), XSD.string)
    _lit(ba, ONTOBPR.applicationType,      app.get("application_type"),     XSD.string)
    is_ret = app.get("is_retrospective")
    if isinstance(is_ret, bool):
        _lit(ba, ONTOBPR.isRetrospective, is_ret, XSD.boolean)
    _lit(ba, ONTOBPR.buildingAuthorityName, app.get("authority_name"), XSD.string)
    _lit(ba, ONTOBPR.siteAddress,          app.get("site_address"),   XSD.string)

    g.add((bldg, RDF.type, ONTOBPR.Building))
    _lit(bldg, ONTOBPR.buildingUseType, bld.get("building_use_type"), XSD.string)

    storeys = bld.get("number_of_storeys")
    if isinstance(storeys, (int, float)):
        _lit(bldg, ONTOBPR.numberOfStoreys, int(storeys), XSD.integer)

    return g
