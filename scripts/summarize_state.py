#!/usr/bin/env python3
"""Local state summary; intentionally no remote actions."""
import json,csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for rel in ['state/STATE.json','state/reconciliation.json']:
 d=json.loads((root/rel).read_text()); print(rel,json.dumps(d,ensure_ascii=False,indent=2))
p=root/'results/INDEX.csv'
with p.open() as f:
 rows=list(csv.DictReader(f))
print('indexed runs:',len(rows))
for r in rows[-10:]: print(r.get('run_id'),r.get('status'),r.get('validity'))
