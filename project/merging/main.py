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


def resolve_config_path(config, path_value):
    config_dir = Path(config.get("_config_dir", Path.cwd())).resolve()
    path = Path(path_value)
    return path if path.is_absolute() else (config_dir / path).resolve()


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

        # Keep query vectors in the same feature space as the retrieval index.
        # This matters for local_bow/ngram modes where the fitted vocabulary
        # differs from the raw document TF-IDF space used by Block 1.
        if self.retrieval_ranking.vectorizer is not None:
            self.query_processor.tfidf_vectorizer = self.retrieval_ranking.vectorizer
        
        # Fit TF-IDF for query processor if needed
        self.logger.info("Processing queries...")
        if self.query_processor.tfidf_vectorizer is None:
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
        summary_metrics, metric_curves = self.evaluate_rankings(rankings, query_ids, qrels)
        
        # Log metrics
        self.logger.log_metrics(summary_metrics)
        
        # Save results
        results = {
            "config": {
                "block1": self.config.get("block1_query_processing", {}),
                "block2": self.config.get("block2_retrieval_mode", {}),
                "block3": self.config.get("block3_ranking_mode", {}),
            },
            "metrics": summary_metrics,
            "metrics_by_k": metric_curves,
            "rankings": {str(qid): ranking for qid, ranking in zip(query_ids, rankings)},
            "timestamp": datetime.now().isoformat()
        }
        
        output_root = self._resolve_config_path(self.config.get("output", {}).get("output_dir", "./output"))
        output_path = output_root / "results.json"
        DataLoader.save_results(results, output_path)
        self.logger.info(f"Results saved to {output_path}")
        
        # Create visualizations
        self.logger.info("Creating visualizations...")
        cutoffs = self.config.get("evaluation", {}).get("cutoffs", [10])
        self.visualize_results(metric_curves, output_path.parent, cutoffs)
        
        self.logger.close()
        return results, summary_metrics
    
    def evaluate_rankings(self, rankings, query_ids, qrels):
        """Evaluate rankings using standard IR metrics"""
        cutoffs = self.config.get("evaluation", {}).get("cutoffs", [10])
        cutoffs = sorted({int(k) for k in cutoffs if int(k) > 0}) or [10]

        metric_names = ["precision", "recall", "map", "ndcg", "mrr"]
        metric_curves = {metric_name: [] for metric_name in metric_names}

        scores_by_cutoff = {
            cutoff: {metric_name: [] for metric_name in metric_names}
            for cutoff in cutoffs
        }
        
        for qid, ranking in zip(query_ids, rankings):
            qid_str = str(qid)
            if qid_str not in qrels:
                continue
            
            relevant_docs = set(str(doc_id) for doc_id in qrels[qid_str].keys())

            for cutoff in cutoffs:
                top_k = ranking[:cutoff]
                relevant_in_top_k = len([d for d in top_k if str(d) in relevant_docs])

                precision = relevant_in_top_k / min(cutoff, len(top_k)) if top_k else 0
                recall = relevant_in_top_k / len(relevant_docs) if relevant_docs else 0

                ap = 0
                for i, doc_id in enumerate(top_k, 1):
                    if str(doc_id) in relevant_docs:
                        precision_at_i = sum(1 for j in range(i) if str(top_k[j]) in relevant_docs) / i
                        ap += precision_at_i
                ap = ap / min(cutoff, len(relevant_docs)) if relevant_docs else 0

                mrr = 0
                for i, doc_id in enumerate(top_k, 1):
                    if str(doc_id) in relevant_docs:
                        mrr = 1 / i
                        break

                idcg = sum(1 / np.log2(i + 1) for i in range(1, min(cutoff, len(relevant_docs)) + 1))
                dcg = sum(1 / np.log2(i + 1) for i, doc_id in enumerate(top_k, 1) if str(doc_id) in relevant_docs)
                ndcg = dcg / idcg if idcg > 0 else 0

                scores_by_cutoff[cutoff]["precision"].append(precision)
                scores_by_cutoff[cutoff]["recall"].append(recall)
                scores_by_cutoff[cutoff]["map"].append(ap)
                scores_by_cutoff[cutoff]["mrr"].append(mrr)
                scores_by_cutoff[cutoff]["ndcg"].append(ndcg)

        for cutoff in cutoffs:
            for metric_name in metric_names:
                values = scores_by_cutoff[cutoff][metric_name]
                metric_curves[metric_name].append(float(np.mean(values)) if values else 0.0)

            self.logger.log_metrics(
                {
                    "k": cutoff,
                    "precision": metric_curves["precision"][-1],
                    "recall": metric_curves["recall"][-1],
                    "map": metric_curves["map"][-1],
                    "ndcg": metric_curves["ndcg"][-1],
                    "mrr": metric_curves["mrr"][-1],
                },
                step=cutoff,
            )

        summary_cutoff = 10 if 10 in cutoffs else cutoffs[-1]
        summary_index = cutoffs.index(summary_cutoff)
        summary_metrics = {
            f"precision@{summary_cutoff}": metric_curves["precision"][summary_index],
            f"recall@{summary_cutoff}": metric_curves["recall"][summary_index],
            f"map@{summary_cutoff}": metric_curves["map"][summary_index],
            f"ndcg@{summary_cutoff}": metric_curves["ndcg"][summary_index],
            f"mrr@{summary_cutoff}": metric_curves["mrr"][summary_index],
        }

        self.logger.info(f"Metrics: {summary_metrics}")
        return summary_metrics, metric_curves

    def _resolve_config_path(self, path_value):
        config_dir = Path(self.config.get("_config_dir", Path.cwd())).resolve()
        path = Path(path_value)
        return path if path.is_absolute() else (config_dir / path).resolve()

    def visualize_results(self, metric_curves, output_dir, cutoffs):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metric_names = ["precision", "recall", "map", "ndcg", "mrr"]
            fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
            axes = axes.flatten()

            for index, metric_name in enumerate(metric_names):
                ax = axes[index]
                ax.plot(cutoffs, metric_curves[metric_name], marker="o", linewidth=2, color="#1f77b4")
                ax.set_title(f"{metric_name.upper()} vs k")
                ax.set_xlabel("k")
                ax.set_ylabel("Score")
                ax.set_ylim(0, 1)
                ax.set_xticks(cutoffs)
                ax.grid(True, alpha=0.25)

            axes[-1].axis("off")
            axes[-1].text(0.0, 1.0, "Metric curves logged to WandB", va="top", family="monospace")

            fig.suptitle("IR Metrics Across Cutoffs", fontsize=16)
            
            plot_path = output_dir / "metrics_plot.png"
            plt.savefig(plot_path, dpi=100)

            self.logger.info(f"Plot saved to {plot_path}")
            plt.close(fig)
            
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
    config_path = Path(args.config).resolve()
    with open(config_path, 'r') as f:
        config = json.load(f)
    config_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    config["_config_dir"] = str(config_root)
    
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
    config_output = resolve_config_path(config, config.get("output", {}).get("output_dir", "./output")) / "config.json"
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
