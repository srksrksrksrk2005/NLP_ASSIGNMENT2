"""
Experiment Runner
Allows easy testing of different mode combinations
"""

import json
import subprocess
import sys
from pathlib import Path
from itertools import product
from datetime import datetime


def _load_run_metrics(config_path):
    """Load metrics from the latest pipeline results file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    output_dir = Path(config.get("output", {}).get("output_dir", "./output"))
    results_path = output_dir / "results.json"
    if not results_path.exists():
        return None

    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        return results.get("metrics")
    except Exception:
        return None


def _format_run_label(result):
    """Create a compact label for a run configuration."""
    reduction_label = "red" if result.get("reduction") else "no-red"
    return (
        f"{result.get('block1_mode')} | {reduction_label} | "
        f"{result.get('block2_retrieval')} | {result.get('block3_ranking')}"
    )


def plot_grid_search_results(results_summary, interactive=True):
    """Show an interactive summary window for all grid-search configurations."""
    successful_runs = [result for result in results_summary if result.get("success") and result.get("metrics")]
    if not successful_runs:
        print("No successful runs with metrics available for plotting.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not import matplotlib for plotting: {exc}")
        return

    metric_names = ["precision@10", "recall@10", "map@10", "ndcg@10", "mrr@10"]
    run_labels = [_format_run_label(result) for result in successful_runs]
    run_indices = list(range(1, len(successful_runs) + 1))

    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    axes = axes.flatten()

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        metric_values = [float(result["metrics"].get(metric_name, 0.0)) for result in successful_runs]
        ax.plot(run_indices, metric_values, marker="o", linewidth=1.5, color="#1f77b4")
        ax.set_title(metric_name)
        ax.set_xlabel("Run")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

        if len(run_indices) <= 20:
            ax.set_xticks(run_indices)
            ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=8)
        else:
            step = max(1, len(run_indices) // 10)
            ax.set_xticks(run_indices[::step])
            ax.set_xticklabels(run_labels[::step], rotation=45, ha="right", fontsize=8)

    # Hide the unused subplot and use it for a compact legend-style summary.
    axes[-1].axis("off")
    metrics_text = "\n".join(
        f"{i}. {label}" for i, label in enumerate(run_labels[:12], start=1)
    )
    if len(run_labels) > 12:
        metrics_text += f"\n... and {len(run_labels) - 12} more runs"
    axes[-1].text(0.0, 1.0, metrics_text or "No runs to display", va="top", family="monospace")

    fig.suptitle("Grid Search Metrics Across Run Configurations", fontsize=16)

    if interactive:
        print("Launching interactive plot window...")
        plt.show()
    else:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / f"grid_search_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(plot_path, dpi=120)
        print(f"Saved plot to: {plot_path}")
        plt.close(fig)


def run_experiment(block1_mode, block1_reduce, block2_retrieval, block3_ranking, 
                   use_wandb=False, config_path="configs/default_config.json"):
    """Run a single experiment with given parameters"""
    
    cmd = [sys.executable, "main.py", "--config", config_path]
    
    cmd.extend(["--block1-mode", block1_mode])
    if block1_reduce:
        cmd.append("--block1-reduce")
    cmd.extend(["--block2-retrieval", block2_retrieval])
    cmd.extend(["--block3-ranking", block3_ranking])
    
    if use_wandb:
        cmd.append("--wandb")
    
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    success = result.returncode == 0
    metrics = _load_run_metrics(config_path) if success else None

    return {
        "block1_mode": block1_mode,
        "reduction": block1_reduce,
        "block2_retrieval": block2_retrieval,
        "block3_ranking": block3_ranking,
        "success": success,
        "metrics": metrics,
    }


def run_grid_search(block1_modes=None, block2_retrievals=None, block3_rankings=None,
                    reduction_options=None, use_wandb=False, config_path="configs/default_config.json",
                    interactive_plot=False):
    """
    Run grid search over multiple mode combinations
    
    Args:
        block1_modes: List of Block 1 expansion modes
        block2_retrievals: List of Block 2 retrieval types
        block3_rankings: List of Block 3 ranking types
        reduction_options: List of reduction options (True/False)
        use_wandb: Whether to log to WandB
        config_path: Path to config file
    """
    
    block1_modes = block1_modes or ["none", "lsa", "wordnet"]
    block2_retrievals = block2_retrievals or ["tfidf", "ngram"]
    block3_rankings = block3_rankings or ["tfidf", "lsa", "esa"]
    reduction_options = reduction_options or [False]
    
    total_experiments = (len(block1_modes) * len(block2_retrievals) * 
                        len(block3_rankings) * len(reduction_options))
    
    print(f"\n{'='*70}")
    print(f"Grid Search Configuration")
    print(f"{'='*70}")
    print(f"Block 1 modes: {block1_modes}")
    print(f"Block 2 retrievals: {block2_retrievals}")
    print(f"Block 3 rankings: {block3_rankings}")
    print(f"Reduction options: {reduction_options}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*70}\n")
    
    results_summary = []
    experiment_num = 1
    
    for block1_mode, reduce, retrieval, ranking in product(
        block1_modes, reduction_options, block2_retrievals, block3_rankings
    ):
        print(f"\n[{experiment_num}/{total_experiments}] ", end="")
        experiment_result = run_experiment(block1_mode, reduce, retrieval, ranking,
                                           use_wandb=use_wandb, config_path=config_path)

        success = experiment_result["success"]
        result_str = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{result_str}")

        results_summary.append(experiment_result)
        
        experiment_num += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Grid Search Results Summary")
    print(f"{'='*70}")
    
    successful = sum(1 for r in results_summary if r["success"])
    failed = len(results_summary) - successful
    
    print(f"Total: {len(results_summary)} experiments")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results_summary)*100:.1f}%")
    
    print(f"\nDetailed Results:")
    print(f"{'-'*70}")
    for result in results_summary:
        status = "✓" if result["success"] else "✗"
        reduction_label = "yes" if result["reduction"] else "no"
        print(f"{status} Block1={result['block1_mode']:8s} Reduce={reduction_label:3s} " +
              f"Block2={result['block2_retrieval']:10s} Block3={result['block3_ranking']:6s}")
    
    print(f"{'='*70}\n")
    
    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    summary_file = output_dir / f"grid_search_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}\n")

    if interactive_plot:
        plot_grid_search_results(results_summary, interactive=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Runner for Merging Pipeline")
    parser.add_argument("--mode", choices=["single", "grid"], default="single",
                        help="Run mode: single experiment or grid search")
    parser.add_argument("--block1", type=str, default="none",
                        help="Block 1 expansion mode")
    parser.add_argument("--reduce", action="store_true",
                        help="Enable query reduction")
    parser.add_argument("--block2", type=str, default="tfidf",
                        help="Block 2 retrieval type")
    parser.add_argument("--block3", type=str, default="tfidf",
                        help="Block 3 ranking type")
    parser.add_argument("--config", type=str, default="configs/default_config.json",
                        help="Config file path")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--interactive-plot", action="store_true",
                        help="Launch an interactive plot window after grid search")
    
    # Grid search options
    parser.add_argument("--grid-block1", type=str, nargs="+",
                        help="Block 1 modes for grid search")
    parser.add_argument("--grid-block2", type=str, nargs="+",
                        help="Block 2 retrieval types for grid search")
    parser.add_argument("--grid-block3", type=str, nargs="+",
                        help="Block 3 ranking types for grid search")
    parser.add_argument("--grid-reduce", action="store_true",
                        help="Include reduction in grid search")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        print(f"Running single experiment...")
        result = run_experiment(args.block1, args.reduce, args.block2, args.block3,
                                use_wandb=args.wandb, config_path=args.config)
        sys.exit(0 if result["success"] else 1)
    
    else:  # grid search
        print(f"Running grid search...")
        
        # Default grid configurations
        block1_modes = args.grid_block1 or ["none", "lsa", "wordnet", "tfidf"]
        block2_retrievals = args.grid_block2 or ["tfidf", "ngram", "local_bow"]
        block3_rankings = args.grid_block3 or ["tfidf", "lsa", "esa"]
        reduction_options = [False, True] if args.grid_reduce else [False]
        
        run_grid_search(block1_modes, block2_retrievals, block3_rankings,
                       reduction_options, use_wandb=args.wandb,
                       config_path=args.config,
                       interactive_plot=args.interactive_plot)


if __name__ == "__main__":
    main()
