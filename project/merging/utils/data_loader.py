import json
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix


class DataLoader:
    """Loads and manages dataset (queries, documents, qrels)"""
    
    def __init__(self, config):
        self.config = config
        self.dataset_path = Path(config.get("dataset", {}).get("path", "./cranfield"))
        self.queries = None
        self.docs = None
        self.qrels = None
        self.queries_json = None
        self.docs_json = None
    
    def load_dataset(self):
        """Load queries, documents, and relevance judgments"""
        dataset_config = self.config.get("dataset", {})
        queries_file = self.dataset_path / dataset_config.get("queries_file", "cran_queries.json")
        docs_file = self.dataset_path / dataset_config.get("docs_file", "cran_docs.json")
        qrels_file = self.dataset_path / dataset_config.get("qrels_file", "cran_qrels.json")
        
        self.queries_json = json.load(open(queries_file, 'r'))
        self.docs_json = json.load(open(docs_file, 'r'))
        # Qrels in Cranfield are a list of entries; convert to mapping
        qrels_list = json.load(open(qrels_file, 'r'))
        qrels_dict = {}
        for entry in qrels_list:
            qnum = str(entry.get("query_num") or entry.get("query number") or entry.get("query_num"))
            docid = str(entry.get("id"))
            # position or relevance score if available
            rel = entry.get("position", 1)
            if qnum not in qrels_dict:
                qrels_dict[qnum] = {}
            qrels_dict[qnum][docid] = rel
        self.qrels = qrels_dict
        
        self.queries = [item["query"] for item in self.queries_json]
        self.docs = [item["body"] for item in self.docs_json]
        self.query_ids = [item["query number"] for item in self.queries_json]
        self.doc_ids = [item["id"] for item in self.docs_json]
        
        return {
            "queries": self.queries,
            "docs": self.docs,
            "query_ids": self.query_ids,
            "doc_ids": self.doc_ids,
            "qrels": self.qrels
        }
    
    def get_queries(self):
        if self.queries is None:
            self.load_dataset()
        return self.queries, self.query_ids
    
    def get_docs(self):
        if self.docs is None:
            self.load_dataset()
        return self.docs, self.doc_ids
    
    def get_qrels(self):
        if self.qrels is None:
            self.load_dataset()
        return self.qrels
    
    @staticmethod
    def save_results(results, output_path):
        """Save processing results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    @staticmethod
    def load_results(input_path):
        """Load processing results from JSON"""
        with open(input_path, 'r') as f:
            return json.load(f)
