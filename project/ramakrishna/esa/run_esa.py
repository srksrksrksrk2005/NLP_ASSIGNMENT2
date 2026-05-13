import argparse, sys, time
from pathlib import Path

# set up paths so we can import from the main project
CURRENT_DIR = Path(__file__).resolve().parent
RAMAKRISHNA_ROOT = CURRENT_DIR.parent
ASSIGNMENT_ROOT = CURRENT_DIR.parents[2]

for path in [str(ASSIGNMENT_ROOT), str(RAMAKRISHNA_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from esa_experiment import ESAExperiment


def main():
    t0 = time.time()
    
    parser = argparse.ArgumentParser(description="ESA retrieval on Cranfield")
    parser.add_argument("-dataset", default=str(ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("-out_folder", default=str(CURRENT_DIR / "output_esa"))
    parser.add_argument("-segmenter", default="punkt", choices=["naive", "punkt"])
    parser.add_argument("-tokenizer", default="ptb", choices=["naive", "ptb"])

    parser.add_argument("-lsa_components", type=int, default=250)
    parser.add_argument("-hybrid_lsa_components", type=int, default=240)
    parser.add_argument("-tfidf_weight", type=float, default=0.2)
    parser.add_argument("-esa_top_concepts", type=int, default=25)
    parser.add_argument("-esa_min_similarity", type=float, default=0.0)
    
    parser.add_argument("-concept_source", default="cranfield",
                        choices=["cranfield", "20news-technical"])
    parser.add_argument("-concept_limit", type=int, default=0)
    parser.add_argument("-prebuilt_index_path", default="")
    parser.add_argument("-random_state", type=int, default=42)

    parser.add_argument("-max_df", type=float, default=0.9)
    parser.add_argument("-min_df", type=int, default=1)
    parser.add_argument("-tfidf_norm", default="l2", choices=["l1", "l2", "none"])
    parser.add_argument("-ngram_max", type=int, default=2, choices=[1, 2])
    parser.add_argument("-disable_sublinear_tf", action="store_true")
    parser.add_argument("-custom", action="store_true")
    
    args = parser.parse_args()
    if args.tfidf_norm == "none":
        args.tfidf_norm = None

    exp = ESAExperiment(args)

    if args.custom:
        # interactive query mode
        _, docs_json, _ = exp.load_dataset()
        doc_ids = [d["id"] for d in docs_json]
        docs = [d["body"] for d in docs_json]
        proc_docs = exp.preprocess(docs)
        
        query = input("Enter query below\n")
        proc_query = exp.preprocess([query])
        
        concept_docs = concept_ids = concept_source = None
        if not args.prebuilt_index_path:
            concept_docs, concept_ids, concept_source = exp.load_concept_corpus()
            
        ranked, _ = exp.rank_esa(
            proc_docs, doc_ids, proc_query,
            concept_docs=concept_docs, concept_ids=concept_ids,
            concept_source=concept_source,
            prebuilt_index_path=args.prebuilt_index_path or None,
        )
        
        print("\nTop five document IDs:")
        for doc_id in ranked[0][:5]:
            print(doc_id)
            
    else:
        # standard evaluation mode
        results = exp.run_evaluation()
        print("ESA plots saved in:", args.out_folder)
        
        print("\n--- RESULTS @ 10 ---")
        for method in ["tfidf", "lsa", "hybrid", "esa"]:
            m = results["metrics"][method]
            print(f"\n{method.upper()}")
            print(f"P@10:    {m['precision'][-1]:.4f}")
            print(f"R@10:    {m['recall'][-1]:.4f}")
            print(f"F1@10:   {m['fscore'][-1]:.4f}")
            print(f"MAP@10:  {m['map'][-1]:.4f}")
            print(f"nDCG@10: {m['ndcg'][-1]:.4f}")
            print(f"MRR@10:  {m['mrr'][-1]:.4f}")

    print(f"\nTime taken: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
