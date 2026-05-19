"""Pipeline v2 — reforma híbrida Sonata + Hungarian + RF + AI alignment.

Toda a reforma vive aqui, isolada do `app_obb.py` principal. O backend
expõe `/api/analisar_ai_v2` que delega pra `orchestrator.run(...)`.

Frentes:
  B — Sonata (segmentação semântica)        ← sonata_runner.py, sonata_cache.py, class_mapping.py
  C — Hungarian matching                     ← matcher_hungarian.py, matcher_costs.py
  D — Features v2 + RF v2                    ← features_v2.py, rf_router.py
  A — AI Alignment (GeoTransformer)          ← alignment_ai.py, alignment_fallback.py

Subprocesses pra ambientes externos (Sonata em venv próprio, GeoTransformer idem)
via `subprocess.run` com IO em pickle/npy.
"""
