import json
data = json.load(open('results/all_results.json', 'r'))
results = []
for ds in data:
    for bal in data[ds]:
        for m in data[ds][bal]:
            r = data[ds][bal][m]
            results.append((r['accuracy']['mean'], r['f1']['mean'], ds, bal, m))

results.sort(reverse=True)
print("=== TOP 15 BY ACCURACY ===")
for acc, f1, ds, bal, m in results[:15]:
    print(f"acc={acc:.4f}  f1={f1:.4f}  {ds} | {bal:12s} | {m}")

print("\n=== BEST PER DATASET ===")
for ds_name in ['Dataset_A', 'Dataset_B']:
    ds_results = [(r['accuracy']['mean'], r['f1']['mean'], bal, m)
                  for bal in data[ds_name] for m in data[ds_name][bal]]
    ds_results.sort(reverse=True)
    print(f"\n{ds_name}:")
    for acc, f1, bal, m in ds_results[:3]:
        print(f"  acc={acc:.4f}  f1={f1:.4f}  {bal:12s} | {m}")
