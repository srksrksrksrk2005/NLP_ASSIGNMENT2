"""
Verify the expanded limitation test set:
1. Check all Cranfield query numbers exist in cran_queries.json
2. Check all relevance doc IDs exist in cran_qrels.json for the given query
3. Print summary statistics
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAN = os.path.join(BASE, "cranfield")

# Load Cranfield data
with open(os.path.join(CRAN, "cran_queries.json"), encoding="utf-8") as f:
    queries_raw = json.load(f)
queries_map = {q["query number"]: q["query"] for q in queries_raw}

with open(os.path.join(CRAN, "cran_qrels.json"), encoding="utf-8") as f:
    qrels_raw = json.load(f)

# Build qrels lookup: query_num -> set of doc ids
qrels = {}
for entry in qrels_raw:
    qn = int(entry["query_num"])
    did = entry["id"]
    qrels.setdefault(qn, set()).add(did)

# Load our test set
with open(os.path.join(BASE, "project", "limitation_test_set_expanded.json"), encoding="utf-8") as f:
    test_set = json.load(f)

print("=" * 70)
print("VERIFICATION REPORT")
print("=" * 70)

total_cranfield = 0
total_custom = 0
errors = []

for lim_key, lim_data in test_set["limitations"].items():
    cran_qs = lim_data.get("cranfield_queries", [])
    cust_qs = lim_data.get("custom_queries", [])
    n_cran = len(cran_qs)
    n_cust = len(cust_qs)
    total_cranfield += n_cran
    total_custom += n_cust

    print(f"\n--- {lim_key} ---")
    print(f"  Cranfield queries: {n_cran}")
    print(f"  Custom queries:    {n_cust}")
    print(f"  TOTAL:             {n_cran + n_cust}")

    if n_cran + n_cust < 20:
        errors.append(f"[WARN] {lim_key}: only {n_cran + n_cust} queries (need >= 20)")

    # Verify Cranfield queries
    for cq in cran_qs:
        qn = cq["query_number"]
        # Check query exists
        if qn not in queries_map:
            errors.append(f"[ERROR] {lim_key}: query {qn} NOT found in cran_queries.json")
        else:
            # Verify query text matches (first 50 chars)
            expected_start = queries_map[qn][:50].strip()
            actual_start = cq["query"][:50].strip()
            if expected_start != actual_start:
                errors.append(f"[WARN] {lim_key}: query {qn} text mismatch. Expected: '{expected_start}...' Got: '{actual_start}...'")

        # Check relevance docs
        rel_docs = cq.get("relevant_docs", [])
        if qn in qrels:
            for doc_id in rel_docs:
                if doc_id not in qrels[qn]:
                    errors.append(f"[ERROR] {lim_key}: query {qn}, doc {doc_id} NOT in qrels")
        else:
            if len(rel_docs) > 0:
                errors.append(f"[WARN] {lim_key}: query {qn} has no qrels but {len(rel_docs)} docs listed")

    # Basic check on custom queries
    for cust in cust_qs:
        if len(cust.get("query", "")) < 10:
            errors.append(f"[WARN] {lim_key}: custom query '{cust.get('custom_id','')}' too short")

print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"{'=' * 70}")
print(f"Total limitations:      {len(test_set['limitations'])}")
print(f"Total Cranfield queries: {total_cranfield}")
print(f"Total custom queries:    {total_custom}")
print(f"Grand total queries:     {total_cranfield + total_custom}")
print(f"Errors/warnings found:   {len(errors)}")

if errors:
    print(f"\nDETAILED ISSUES:")
    for e in errors:
        print(f"  {e}")
else:
    print(f"\n*** ALL CHECKS PASSED ***")
