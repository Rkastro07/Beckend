from __future__ import annotations
import json,sys
from collections import Counter
from pathlib import Path

proposals=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
truth=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
by_wall={wall["wall_id"]:wall["candidates"] for wall in proposals["walls"]}
expected=matched=proposed=0; rows=[]
for wall_id,target in truth["walls"].items():
    counts=Counter(c["type"] for c in by_wall.get(wall_id,[]) if c["status"]=="proposed")
    row={"wall_id":wall_id,"expected":{},"proposed":{},"matched":{}}
    for kind in ("door","window"):
        want=int(target.get(kind,0)); got=int(counts.get(kind,0)); hit=min(want,got)
        expected+=want; proposed+=got; matched+=hit
        row["expected"][kind]=want; row["proposed"][kind]=got; row["matched"][kind]=hit
    rows.append(row)
precision=matched/proposed if proposed else 0.; recall=matched/expected if expected else 0.
result={"matched":matched,"proposed":proposed,"expected":expected,"precision":round(precision,4),"recall":round(recall,4),"rows":rows}
print(json.dumps(result,ensure_ascii=False,indent=2))
