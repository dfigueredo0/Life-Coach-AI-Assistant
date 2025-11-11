import json
from pathlib import Path
from random import shuffle

IN = Path('data/sft/samples.jsonl')
OUT = Path('data/rm/rm_train.jsonl')

OUT.parent.mkdir(parents=True, exist_ok=True)

items = []
with open(IN, 'r', encoding='utf-8') as f:
    for line in f:
        ex = json.loads(line)
        prompt = ex.get('prompt') or ex.get('input') or ex.get('goal_description')
        
        if 'responses' in ex and len(ex['responses']) >= 2:
            r = ex['response']
            items.append({  
                "prompt": prompt,
                "chosen": r[0],
                "rejected": r[1]
            })
        elif 'model' in ex and 'gold' in ex:
            items.append({
                "prompt": prompt,
                "chose": ex['gold'],
                'rejected': ex['model']
            })

with open(OUT, "w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item) + "\n")

print(f"Wrote {len(items)} RM pairs → {OUT}")