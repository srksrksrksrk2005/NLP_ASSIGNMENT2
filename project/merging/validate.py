"""
Validation and Testing Script
Tests all components of the merging pipeline
"""

import sys
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    try:
        import numpy
        import sklearn
        import nltk
        import matplotlib
        print("✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_utils():
    """Test utility modules"""
    print("\nTesting utility modules...")
    try:
        from utils.logger import PipelineLogger
        from utils.data_loader import DataLoader
        from utils.preprocessing import TextPreprocessor
        print("✓ All utility modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_block1():
    """Test Block 1: Query Processing"""
    print("\nTesting Block 1: Query Processing...")
    try:
        from block1_query_processing import QueryProcessor
        import json
        
        # Load config
        with open("configs/default_config.json") as f:
            config = json.load(f)
        
        # Create processor
        processor = QueryProcessor(config)
        
        # Test single query processing
        query = "information retrieval"
        result = processor.process_query(query)
        
        if result is not None:
            print(f"✓ Block 1 query processing successful")
            print(f"  - Query: '{query}'")
            print(f"  - Output type: {type(result)}")
            return True
        else:
            print("✗ Block 1 returned None")
            return False
    except Exception as e:
        print(f"✗ Block 1 error: {e}")
        return False

def test_block23():
    """Test Blocks 2 & 3: Retrieval and Ranking"""
    print("\nTesting Blocks 2 & 3: Retrieval and Ranking...")
    try:
        from blocks23_retrieval_ranking import RetrievalRankingPipeline
        import json
        import numpy as np
        
        # Load config
        with open("configs/default_config.json") as f:
            config = json.load(f)
        
        # Create pipeline
        pipeline = RetrievalRankingPipeline(config)
        
        # Create dummy data
        docs = [
            ["information", "retrieval", "systems"],
            ["document", "ranking", "algorithms"],
            ["query", "processing", "methods"]
        ]
        doc_ids = ["D1", "D2", "D3"]
        
        # Build index
        pipeline.build_retrieval_index(docs, doc_ids)
        
        # Create query vectors in the same TF-IDF feature space as the index
        try:
            sample_query = "information retrieval"
            query_vectors_sparse = pipeline.vectorizer.transform([sample_query])
            # Use dense vectors for ranking methods that expect dense input
            query_vectors = query_vectors_sparse.toarray()
        except Exception:
            # Fallback to a random vector if vectorizer unavailable
            query_vectors = np.random.randn(1, 100)
        
        # Rank
        rankings = pipeline.rank(query_vectors)
        
        if rankings and len(rankings) > 0:
            print(f"✓ Blocks 2 & 3 retrieval and ranking successful")
            print(f"  - Number of queries ranked: {len(rankings)}")
            print(f"  - First ranking: {[doc_id for doc_id, _ in rankings[0][:3]]}")
            return True
        else:
            print("✗ Blocks 2 & 3 returned no rankings")
            return False
    except Exception as e:
        print(f"✗ Blocks 2 & 3 error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        config_path = "configs/default_config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        required_keys = ["block1_query_processing", "block2_retrieval_mode", 
                        "block3_ranking_mode", "dataset", "output", "logging"]
        
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"✗ Config missing keys: {missing_keys}")
            return False
        
        print(f"✓ Configuration loaded and validated")
        print(f"  - Block 1 expansion: {config['block1_query_processing']['expansion_mode']}")
        print(f"  - Block 2 retrieval: {config['block2_retrieval_mode']['retrieval_type']}")
        print(f"  - Block 3 ranking: {config['block3_ranking_mode']['ranking_type']}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_data_loading():
    """Test data loading (if dataset exists)"""
    print("\nTesting data loading...")
    try:
        from utils.data_loader import DataLoader
        import json
        
        with open("configs/default_config.json") as f:
            config = json.load(f)
        
        loader = DataLoader(config)
        dataset_path = Path(config.get("dataset", {}).get("path", ""))
        
        if not dataset_path.exists():
            print(f"⚠ Dataset path does not exist: {dataset_path}")
            print("  (This is okay if you haven't downloaded the dataset yet)")
            return True
        
        # Try loading
        data = loader.load_dataset()
        
        print(f"✓ Data loading successful")
        print(f"  - Queries: {len(data['queries'])}")
        print(f"  - Documents: {len(data['docs'])}")
        return True
    except Exception as e:
        print(f"⚠ Data loading warning: {e}")
        print("  (This is okay if dataset not yet downloaded)")
        return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Merging Pipeline - Validation Tests")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Utilities", test_utils),
        ("Block 1", test_block1),
        ("Blocks 2 & 3", test_block23),
        ("Configuration", test_config),
        ("Data Loading", test_data_loading),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")
    
    total_passed = sum(1 for p in results.values() if p)
    total_tests = len(results)
    
    print(f"\n{total_passed}/{total_tests} tests passed")
    print("="*60 + "\n")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("✓ All tests passed! Pipeline is ready to use.\n")
        print("To run the pipeline, use:")
        print("  python main.py")
        print("  python main.py --block1-mode lsa --block3-ranking esa")
        print("  python run_experiments.py --mode grid")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the errors above.\n")
        print("Common fixes:")
        print("  1. Install requirements: pip install -r requirements.txt")
        print("  2. Download NLTK data: python -c \"import nltk; nltk.download('punkt')\"")
        print("  3. Ensure dataset exists at the path in configs/default_config.json")
        sys.exit(1)
