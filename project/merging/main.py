"""
Main Pipeline Runner
Orchestrates Block 1 (Query Processing), Blocks 2&3 (Retrieval & Ranking)
with evaluation, visualization, and wandb logging
"""

import json
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import PipelineLogger
from utils.data_loader import DataLoader
from block1_query_processing import QueryProcessor
from blocks23_retrieval_ranking import RetrievalRankingPipeline


class MergingPipeline:
    """Main pipeline combining all blocks"""
    
    def __init__(self, config_source):
        """Initialize the pipeline"""
        if isinstance(config_source, (str, Path)):
            config_path = Path(config_source)
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.config_source = str(config_path)
        else:
            self.config = config_source
            self.config_source = "<in-memory-config>"
        
        self.logger = PipelineLogger(self.config, log_name="MergingPipeline")
        self.data_loader = DataLoader(self.config)
        self.query_processor = QueryProcessor(self.config)
        self.retrieval_ranking = RetrievalRankingPipeline(self.config)
        
        self.logger.info(f"Pipeline initialized with config from {self.config_source}")
        self.logger.info(f"Block 1 (Query): mode={self.query_processor.expansion_mode}, reduce={self.query_processor.reduction_enabled}")
        self.logger.info(f"Block 2 (Retrieval): {self.retrieval_ranking.retrieval_type}")
        self.logger.info(f"Block 3 (Ranking): {self.retrieval_ranking.ranking_type}")
    
    def run(self):
        """Run the complete pipeline"""
        self.logger.info("Loading dataset...")
        dataset = self.data_loader.load_dataset()
        
        queries = dataset["queries"]
        docs = dataset["docs"]
        query_ids = dataset["query_ids"]
        doc_ids = dataset["doc_ids"]
        qrels = dataset["qrels"]
        
        self.logger.info(f"Dataset loaded: {len(queries)} queries, {len(docs)} documents")
        
        # Preprocess documents and queries
        self.logger.info("Preprocessing documents...")
        preprocessed_docs = [doc.split() for doc in docs]
        
        # Build retrieval index
        self.logger.info("Building retrieval index...")
        self.retrieval_ranking.build_retrieval_index(preprocessed_docs, doc_ids)
        
        # Fit TF-IDF for query processor if needed
        self.logger.info("Processing queries...")
        self.query_processor.fit_tfidf(preprocessed_docs)
        
        # Process queries (Block 1)
        self.logger.info(f"Applying Block 1 processing (expansion={self.query_processor.expansion_mode}, reduce={self.query_processor.reduction_enabled})...")
        processed_queries = self.query_processor.process_batch(queries, preprocessed_docs, return_vectors=True)
        
        # Rank documents (Blocks 2 & 3)
        self.logger.info(f"Ranking with Block 2/3 (retrieval={self.retrieval_ranking.retrieval_type}, ranking={self.retrieval_ranking.ranking_type})...")
        ranked_results = self.retrieval_ranking.rank(processed_queries)
        
        # Convert rankings to doc IDs only
        rankings = []
        for query_results in ranked_results:
            doc_ranking = [doc_id for doc_id, score in query_results]
            rankings.append(doc_ranking)
        
        # Evaluate
        self.logger.info("Evaluating results...")
        metrics = self.evaluate_rankings(rankings, query_ids, qrels)
        
        # Log metrics
        self.logger.log_metrics(metrics)
        
        # Save results
        results = {
            "config": {
                "block1": self.config.get("block1_query_processing", {}),
                "block2": self.config.get("block2_retrieval_mode", {}),
                "block3": self.config.get("block3_ranking_mode", {}),
            },
            "metrics": metrics,
            "rankings": {str(qid): ranking for qid, ranking in zip(query_ids, rankings)},
            "timestamp": datetime.now().isoformat()
        }
        
        output_path = Path(self.config.get("output", {}).get("output_dir", "./output")) / "results.json"
        DataLoader.save_results(results, output_path)
        self.logger.info(f"Results saved to {output_path}")
        
        # Create visualizations
        self.logger.info("Creating visualizations...")
        self.visualize_results(metrics, output_path.parent)
        
        self.logger.close()
        return results, metrics
    
    def evaluate_rankings(self, rankings, query_ids, qrels):
        """Evaluate rankings using standard IR metrics"""
        metrics = {
            "precision@10": 0,
            "recall@10": 0,
            "map@10": 0,
            "ndcg@10": 0,
            "mrr@10": 0
        }
        
        precision_scores = []
        recall_scores = []
        ap_scores = []
        ndcg_scores = []
        mrr_scores = []
        
        for qid, ranking in zip(query_ids, rankings):
            qid_str = str(qid)
            if qid_str not in qrels:
                continue
            
            relevant_docs = set(str(doc_id) for doc_id in qrels[qid_str].keys())
            
            # Precision@10
            top_10 = ranking[:10]
            relevant_in_top_10 = len([d for d in top_10 if str(d) in relevant_docs])
            precision = relevant_in_top_10 / min(10, len(top_10)) if top_10 else 0
            precision_scores.append(precision)
            
            # Recall@10
            recall = relevant_in_top_10 / len(relevant_docs) if relevant_docs else 0
            recall_scores.append(recall)
            
            # MAP@10
            ap = 0
            for i, doc_id in enumerate(top_10, 1):
                if str(doc_id) in relevant_docs:
                    precision_at_i = sum(1 for j in range(i) if str(ranking[j]) in relevant_docs) / i
                    ap += precision_at_i
            ap = ap / min(10, len(relevant_docs)) if relevant_docs else 0
            ap_scores.append(ap)
            
            # MRR@10
            mrr = 0
            for i, doc_id in enumerate(top_10, 1):
                if str(doc_id) in relevant_docs:
                    mrr = 1 / i
                    break
            mrr_scores.append(mrr)
            
            # NDCG@10
            idcg = sum(1 / np.log2(i + 1) for i in range(1, min(11, len(relevant_docs) + 1)))
            dcg = sum(1 / np.log2(i + 1) for i, doc_id in enumerate(top_10, 1) if str(doc_id) in relevant_docs)
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        if precision_scores:
            metrics["precision@10"] = float(np.mean(precision_scores))
            metrics["recall@10"] = float(np.mean(recall_scores))
            metrics["map@10"] = float(np.mean(ap_scores))
            metrics["mrr@10"] = float(np.mean(mrr_scores))
            metrics["ndcg@10"] = float(np.mean(ndcg_scores))
        
        self.logger.info(f"Metrics: {metrics}")
        return metrics
    
    def visualize_results(self, metrics, output_dir):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Metrics bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            ax.bar(metric_names, metric_values, color='steelblue')
            ax.set_ylabel("Score")
            ax.set_title("IR Metrics")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            
            plot_path = output_dir / "metrics_plot.png"
            plt.savefig(plot_path, dpi=100)
            plt.close()
            
            self.logger.info(f"Plot saved to {plot_path}")
            self.logger.log_plot(fig, "metrics_plot")
            
        except Exception as e:
            self.logger.warning(f"Could not create visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description="Merging Pipeline: Block 1 + Blocks 2&3")
    parser.add_argument("--config", type=str, default="configs/default_config.json",
                        help="Path to config file")
    parser.add_argument("--block1-mode", type=str, choices=["lsa", "esa", "tfidf", "wordnet", "word2vec", "none"],
                        help="Override Block 1 expansion mode")
    parser.add_argument("--block1-reduce", action="store_true",
                        help="Enable Block 1 query reduction")
    parser.add_argument("--block2-retrieval", type=str, choices=["tfidf", "ngram", "local_bow"],
                        help="Override Block 2 retrieval type")
    parser.add_argument("--block3-ranking", type=str, choices=["tfidf", "lsa", "esa"],
                        help="Override Block 3 ranking type")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    
    args = parser.parse_args()
    
    # Load and update config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.block1_mode:
        config["block1_query_processing"]["expansion_mode"] = args.block1_mode
    if args.block1_reduce:
        config["block1_query_processing"]["reduction_enabled"] = True
    if args.block2_retrieval:
        config["block2_retrieval_mode"]["retrieval_type"] = args.block2_retrieval
    if args.block3_ranking:
        config["block3_ranking_mode"]["ranking_type"] = args.block3_ranking
    if args.wandb:
        config["logging"]["use_wandb"] = True
    
    # Save updated config
    config_output = Path(config.get("output", {}).get("output_dir", "./output")) / "config.json"
    config_output.parent.mkdir(parents=True, exist_ok=True)
    with open(config_output, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run pipeline
    print(f"\n{'='*60}")
    print(f"Running Merging Pipeline")
    print(f"{'='*60}")
    print(f"Block 1 (Query): expansion_mode={config['block1_query_processing']['expansion_mode']}, reduce={config['block1_query_processing']['reduction_enabled']}")
    print(f"Block 2 (Retrieval): {config['block2_retrieval_mode']['retrieval_type']}")
    print(f"Block 3 (Ranking): {config['block3_ranking_mode']['ranking_type']}")
    print(f"{'='*60}\n")
    
    pipeline = MergingPipeline(config)
    results, metrics = pipeline.run()
    
    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"{'='*60}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
