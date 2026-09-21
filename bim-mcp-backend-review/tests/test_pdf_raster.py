import math
from unittest.mock import MagicMock

import pytest

from plantatobim.pdf_raster import bounded_scale, render_first_page, MAX_PIXELS, MAX_SIDE


@pytest.mark.parametrize("size", [(2383.92, 3370.44), (3370.44, 2383.92),
                                     (10000, 10000), (1e6, 1), (595, 842)])
def test_raster_budget_and_proportions(size):
    scale = bounded_scale(*size)
    w, h = [math.ceil(side * scale) for side in size]
    assert max(w, h) <= MAX_SIDE
    assert w * h <= MAX_PIXELS
    assert scale <= 3
    assert abs(w - size[0] * scale) < 1
    assert abs(h - size[1] * scale) < 1


@pytest.mark.parametrize("size", [(0, 1), (-1, 1), (float('nan'), 1), (1, float('inf'))])
def test_invalid_dimensions(size):
    with pytest.raises(ValueError):
        bounded_scale(*size)


def test_small_pages_keep_existing_resolution():
    assert bounded_scale(595, 842) == 3


def test_resources_close_even_when_save_fails(monkeypatch, tmp_path):
    import pypdfium2
    document, page, bitmap, image = [MagicMock() for _ in range(4)]
    document.__len__.return_value = 1
    document.__getitem__.return_value = page
    page.get_size.return_value = (2383.92, 3370.44)
    page.render.return_value = bitmap
    bitmap.to_pil.return_value.convert.return_value = image
    image.save.side_effect = OSError("disk full")
    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _: document)
    with pytest.raises(OSError):
        render_first_page(tmp_path / 'in.pdf', tmp_path / 'out.png')
    for resource in (document, page, bitmap, image):
        resource.close.assert_called_once()


def test_wall_masks_expand_individually():
    import numpy as np
    from experiments.plant2bim.run_pre_wall_opening_pipeline import predict_wall_segmentation
    model = MagicMock()
    result = MagicMock()
    result.masks.data.cpu.return_value.numpy.return_value = np.ones((2, 8, 8), dtype=np.float32)
    result.boxes.conf.cpu.return_value.numpy.return_value = [0.8, 0.9]
    model.predict.return_value = [result]
    mask, info = predict_wall_segmentation(model, np.zeros((80, 100, 3), dtype=np.uint8),
                                          image_size=960, confidence=0.15, device='cpu')
    assert model.predict.call_args.kwargs['retina_masks'] is False
    assert mask.shape == (80, 100)
    assert np.all(mask == 255)
    assert info['instance_count'] == 2


def test_wall_masks_remove_letterbox_padding():
    import numpy as np
    from experiments.plant2bim.run_pre_wall_opening_pipeline import predict_wall_segmentation
    model, result = MagicMock(), MagicMock()
    masks = np.zeros((1, 8, 8), dtype=np.float32)
    masks[:, 2:6] = 1  # 2:1 drawing inside square inference canvas
    result.masks.data.cpu.return_value.numpy.return_value = masks
    result.boxes.conf.cpu.return_value.numpy.return_value = [0.9]
    model.predict.return_value = [result]
    mask, _ = predict_wall_segmentation(model, np.zeros((40, 80, 3), dtype=np.uint8),
                                       image_size=960, confidence=0.15, device='cpu')
    assert np.all(mask == 255)


def test_killed_detector_reports_exit_code_without_claiming_oom(monkeypatch, tmp_path):
    import subprocess
    import plantatobim.pre_wall_opening_import as adapter
    executable = tmp_path / 'exists'
    executable.touch()
    monkeypatch.setattr(adapter, '_runtime_path', lambda *_: executable)
    monkeypatch.setattr(adapter.subprocess, 'run', lambda *a, **k:
                        subprocess.CompletedProcess(a, -9, '', ''))
    with pytest.raises(adapter.PreWallOpeningError, match=r'código -9.*SIGKILL.*possível'):
        adapter.run_pre_wall_pipeline(executable, tmp_path / 'out', canvas_width_m=20)
