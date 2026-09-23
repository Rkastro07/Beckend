"""Paid local pilot: heuristic candidates reviewed once by GPT-6 Sol low.

This stage is intentionally separate from the production Astra direct stage.
It is selected only by the explicit local pilot flag after payment approval.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from .gpt_plan_assistant import GptPlanAssistant, OpenAIResponsesClient, build_candidate_model
from .pre_wall_opening_import import pre_wall_image_to_editor_model


class _RecordedSolClient(OpenAIResponsesClient):
    def __init__(self, *, artifact_dir: Path) -> None:
        super().__init__(
            api_key=os.environ.get("OPENAI_API_KEY2") or os.environ.get("OPENAI_API_KEY"),
            model="gpt-6-sol",
            timeout_seconds=float(os.environ.get("OPENAI_PLAN_TIMEOUT_SECONDS", "1200")),
        )
        self.artifact_dir = artifact_dir
        self.max_output_tokens = 32000

    def structured(self, **kwargs):
        kwargs["reasoning_effort"] = "low"
        kwargs["artifact_dir"] = self.artifact_dir
        return super().structured(**kwargs)


def estimate_sol_cost(usage: dict[str, Any]) -> dict[str, Any]:
    """Standard-tier estimate, not a statement of the invoiced amount."""
    total = int(usage.get("input_tokens") or 0)
    details = usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    writes = int(details.get("cache_write_tokens") or 0)
    uncached = max(0, total - cached - writes)
    output = int(usage.get("output_tokens") or 0)
    return {
        "usd_estimate": round((uncached * 2 + cached * 0.2 + writes * 2.5 + output * 10) / 1_000_000, 6),
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-6-sol",
        "uncached_input_tokens": uncached,
        "cached_input_tokens": cached,
        "cache_write_tokens": writes,
        "output_tokens": output,
    }


class SolLowReviewStage:
    """Match the paid stage contract without changing the live Astra stage."""

    def analyze(
        self,
        image_path: Path,
        *,
        canvas_width_m: float,
        original_name: str,
        user_message: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        artifact_dir = Path(image_path).parent / "sol_low_review"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        detector_started = time.perf_counter()
        detector = pre_wall_image_to_editor_model(
            image_path,
            artifact_dir / "detector",
            canvas_width_m=canvas_width_m,
            metric_refinement=True,
        )
        detector_seconds = time.perf_counter() - detector_started
        client = _RecordedSolClient(artifact_dir=artifact_dir / "api")
        review, metadata = GptPlanAssistant(client=client).review_model(
            detector,
            "Revise a geometria pela planta. Preserve candidatos corretos, corrija falsos "
            "positivos e acrescente paredes ou aberturas visíveis que faltaram. "
            "Não invente elementos por simetria; marque dúvidas em unresolved. "
            + user_message[:1500],
        )
        candidate = build_candidate_model(detector, review)
        candidate["nome"] = Path(original_name).stem
        candidate["astra_editor"] = {
            "needs_human_review": True,
        }
        review["_model"] = metadata.get("model") or "gpt-6-sol"
        review["_provider"] = "openai"
        (artifact_dir / "review.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata.update({
            "detector_seconds": round(detector_seconds, 3),
            "stage_seconds": round(time.perf_counter() - started, 3),
            "image_count": 2,
        })
        return candidate, review, metadata
