from __future__ import annotations

import pytest

from plantatobim.raster2seq_import import (
    _raster2seq_cache_paths,
    predictions_to_editor_model,
    run_raster2seq_wsl,
)


def test_predictions_become_aligned_non_destructive_overlay():
    predictions = [
        {
            "id": 1,
            "category_id": 2,
            "segmentation": [[0, 0], [128, 0], [128, 128], [0, 128]],
        },
        {
            "id": 2,
            "category_id": 10,
            "segmentation": [[64, 128], [96, 128]],
        },
        {
            "id": 3,
            "category_id": 9,
            "segmentation": [[128, 32], [128, 64]],
        },
    ]

    model = predictions_to_editor_model(
        predictions,
        image_base64="cG5n",
        source_name="planta.jpg",
        canvas_width_m=20.0,
    )

    reference = model["reference"]
    assert reference["kind"] == "raster2seq"
    assert reference["bounds"] == [0.0, 0.0, 20.0, 20.0]
    assert reference["rooms"][0]["label"] == "Sala"
    assert reference["rooms"][0]["points"] == [
        [0.0, 20.0],
        [10.0, 20.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ]
    assert reference["rooms"][0]["area"] == 100.0
    assert [opening["kind"] for opening in reference["openings"]] == [
        "door",
        "window",
    ]
    assert model["paredes"] == []
    assert model["aberturas"] == []
    assert model["spaces"] == []


def test_invalid_canvas_width_is_rejected():
    with pytest.raises(ValueError, match="largura"):
        predictions_to_editor_model(
            [],
            image_base64="cG5n",
            source_name="planta.png",
            canvas_width_m=0,
        )


def test_invalid_or_too_short_predictions_are_ignored():
    model = predictions_to_editor_model(
        [
            {"category_id": 2, "segmentation": [[1, 1], [2, 2]]},
            {"category_id": 10, "segmentation": [[1, 1]]},
            {"category_id": 9, "segmentation": [[1, 1], ["x", 2], [2, 2]]},
        ],
        image_base64="cG5n",
        source_name="planta.png",
    )

    assert model["reference"]["rooms"] == []
    assert len(model["reference"]["openings"]) == 1


def test_run_raster2seq_reuses_content_addressed_cache(tmp_path, monkeypatch):
    image_path = tmp_path / "planta.jpg"
    image_path.write_bytes(b"same-plan-content")
    monkeypatch.setenv("RASTER2SEQ_CACHE_ROOT", str(tmp_path / "cache"))
    cache_key, json_path, processed_path = _raster2seq_cache_paths(image_path)
    json_path.parent.mkdir(parents=True)
    json_path.write_text(
        '[{"id": "room-1", "category_id": 2, "segmentation": [[0,0],[1,0],[1,1]]}]',
        encoding="utf-8",
    )
    processed_path.write_bytes(b"png-bytes")

    predictions, processed, metadata = run_raster2seq_wsl(
        image_path,
        tmp_path / "output",
    )

    assert predictions[0]["id"] == "room-1"
    assert processed == b"png-bytes"
    assert metadata["cache_hit"] is True
    assert metadata["cache_key"] == cache_key
