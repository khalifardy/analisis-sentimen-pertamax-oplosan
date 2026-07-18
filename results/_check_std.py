import json
data = json.load(open('results/all_results.json', 'r'))
for ds in data:
    for bal in data[ds]:
        for m in data[ds][bal]:
            r = data[ds][bal][m]
            acc = r['accuracy']
            f1 = r['f1']
            print(f"{ds} | {bal:12s} | {m:22s} | acc={acc['mean']:.4f}±{acc['std']:.4f} | f1={f1['mean']:.4f}±{f1['std']:.4f}")
