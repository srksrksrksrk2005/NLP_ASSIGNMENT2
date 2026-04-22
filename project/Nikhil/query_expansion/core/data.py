import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_cranfield_dataset(dataset_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load Cranfield docs, queries and qrels from a dataset directory."""
    base = Path(dataset_dir)
    docs = json.loads((base / "cran_docs.json").read_text(encoding="utf-8"))
    queries = json.loads((base / "cran_queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((base / "cran_qrels.json").read_text(encoding="utf-8"))
    return docs, queries, qrels


def get_docs_ids_and_texts(docs_json: List[Dict]) -> Tuple[List[str], List[str]]:
    doc_ids = [str(item["id"]) for item in docs_json]
    doc_texts = [item.get("body", "") for item in docs_json]
    return doc_ids, doc_texts


def get_query_ids_and_texts(queries_json: List[Dict]) -> Tuple[List[str], List[str]]:
    query_ids = [str(item["query number"]) for item in queries_json]
    query_texts = [item.get("query", "") for item in queries_json]
    return query_ids, query_texts
