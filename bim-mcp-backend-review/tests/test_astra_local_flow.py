from __future__ import annotations

import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pytest

from PIL import Image
import plan_to_bim_free_app as free_app

from plantatobim.astra_local_flow import AstraLocalFlowManager, quote_for_document, _customer_result


def _inspection(area=200):
    return {"status": "verified", "area_m2": area, "scope": "single_floor_first_page",
            "page": 1, "evidence": f"Área construída do pavimento: {area} m²", "message": "Área identificada."}


def test_visual_raw_artifacts_are_not_sent_to_customer():
    model = {"source": {"visual_protocol": "astra-visual-seven-v2",
                        "image_framing_used": True, "geometry_candidates_used": False},
             "gpt_plan": {"_visual_raw": {}, "_visual_manifest": {"local_path": "private"}},
             "astra_direct": {"_visual_raw": {}}, "astra_semantic": {}}
    public = _customer_result(model)
    assert "gpt_plan" not in public
    assert "astra_direct" not in public
    assert "visual_protocol" not in public["source"]
    assert "_visual_manifest" not in str(public)


def _profile(*, width: int = 1600, height: int = 900) -> dict:
    return {
        "page_count": 1,
        "processed_pages": 1,
        "image_width_px": width,
        "image_height_px": height,
        "file_size_bytes": 250_000,
        "heuristic_detector_used": False,
    }


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 32), "white").save(output, format="PNG")
    return output.getvalue()


def _editor_model() -> dict:
    return {
        "ok": True,
        "nome": "sample",
        "engine": "astra-direct-v1",
        "bbox": {"xmin": 0, "ymin": 0, "xmax": 20, "ymax": 10},
        "source": {
            "mode": "astra-direct",
            "geometry_source": "astra-only",
            "heuristic_detector_used": False,
        },
        "paredes": [
            {
                "id": "W-ASTRA-001",
                "ax": 0,
                "ay": 0,
                "bx": 10,
                "by": 0,
                "espessura": 0.15,
                "tipo": "wall",
                "nome": "Parede Astra",
            }
        ],
        "aberturas": [
            {
                "id": "D-ASTRA-001",
                "parede_id": "W-ASTRA-001",
                "tipo": "door",
                "s_centro": 5,
                "largura": 0.8,
            }
        ],
        "laje": {"contorno": [[0, 0], [10, 0], [10, 10]]},
        "astra_editor": {
            "status": "ready-for-manual-review",
            "pipeline_version": "astra-direct-v1",
            "geometry_source": "astra-only",
            "heuristic_detector_used": False,
            "active": {"walls": 1, "openings": 1},
            "pending_review": {"walls": [], "openings": [], "slab": False},
            "excluded": {"walls": [], "openings": []},
            "missing_elements": [],
            "needs_human_review": False,
        },
    }


class FakeStage:
    def analyze(
        self, image_path, *, canvas_width_m, original_name, user_message=""
    ):
        assert image_path.name == "sample.png"
        assert canvas_width_m == 20
        assert original_name == "sample.png"
        assert "Não use detector heurístico" in user_message
        analysis = {
            "message": "Geometria criada diretamente pelo Astra.",
            "changed": True,
            "confidence": 0.94,
            "observations": ["Traços arquitetônicos visíveis."],
            "assumptions": [],
            "walls": [],
            "openings": [],
            "slab_contour": [],
            "unresolved": [],
            "_model": "gpt-6-astra",
        }
        metadata = {
            "provider": "openai",
            "model": "gpt-6-astra",
            "response_id": "resp_test",
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }
        return _editor_model(), analysis, metadata


class FakePaymentGateway:
    configured = True
    webhook_configured = True
    sandbox = True

    def __init__(self):
        self.preferences = []
        self.payments = []

    def create_preference(self, **payload):
        self.preferences.append(payload)
        return {
            "preference_id": f"pref-{payload['job']}",
            "checkout_url": "https://sandbox.mercadopago.com.br/checkout/test",
            "sandbox": True,
        }

    def search_payments(self, external_reference):
        return [
            payment
            for payment in self.payments
            if payment.get("external_reference") == external_reference
        ]

    def get_payment(self, payment_id):
        return next(payment for payment in self.payments if str(payment["id"]) == str(payment_id))

    def verify_webhook(self, **_kwargs):
        return True


class FakeDurableStore:
    def __init__(self):
        self.states = {}
        self.objects = {}

    def save(self, state):
        import copy
        self.states[state["job"]] = copy.deepcopy(state)

    def load(self, job):
        import copy
        state = self.states.get(job)
        return copy.deepcopy(state) if state else None

    def list_for_owner(self, owner_id):
        return [
            self.load(job)
            for job, state in self.states.items()
            if (state.get("owner") or {}).get("id") == owner_id
        ]

    def find_by_external_reference(self, external_reference):
        for state in self.states.values():
            if (state.get("payment") or {}).get("external_reference") == external_reference:
                return self.load(state["job"])
        return None

    def persist_sources(self, job, original_path, image_path):
        self.objects[f"jobs/{job}/input/original.png"] = original_path.read_bytes()
        self.objects[f"jobs/{job}/input/page-1.png"] = image_path.read_bytes()
        return {
            "original_object": f"jobs/{job}/input/original.png",
            "image_object": f"jobs/{job}/input/page-1.png",
        }

    def materialize_sources(self, job, state, root_dir):
        directory = root_dir / job
        directory.mkdir(parents=True, exist_ok=True)
        source = state["source"]
        for path_key, object_key in (
            ("original_path", "original_object"),
            ("image_path", "image_object"),
        ):
            path = directory / "sample.png"
            if not path.is_file():
                path.write_bytes(self.objects[source[object_key]])
            source[path_key] = str(path)
        return state

    def save_result(self, job, editor_model, analysis):
        import copy
        self.objects[f"jobs/{job}/result/editor-model.json"] = copy.deepcopy(editor_model)
        self.objects[f"jobs/{job}/result/analysis.json"] = copy.deepcopy(analysis)
        return {
            "result_object": f"jobs/{job}/result/editor-model.json",
            "analysis_object": f"jobs/{job}/result/analysis.json",
        }

    def load_json(self, object_path):
        import copy
        return copy.deepcopy(self.objects[object_path])


class FakeDispatcher:
    def __init__(self):
        self.tasks = []

    def enqueue(self, job, attempt=0):
        name = f"queues/test/tasks/astra-{job}-{attempt}"
        self.tasks.append((job, attempt, name))
        return name

    def verify_request(self, authorization):
        if authorization != "Bearer valid-test-token":
            raise PermissionError("invalid")
        return {"email": "worker@example.com"}


def _approved_payment(manager: AstraLocalFlowManager, job: str) -> dict:
    state = manager._read(job)
    return {
        "id": "987654321",
        "status": "approved",
        "external_reference": state["payment"]["external_reference"],
        "transaction_amount": state["quote"]["customer_price"],
        "currency_id": "BRL",
    }


def _create_job(manager: AstraLocalFlowManager, tmp_path):
    source = tmp_path / "sample.png"
    source.write_bytes(_png_bytes())
    return manager.create_job(
        job="0123456789",
        original_name="sample.png",
        original_path=source,
        image_path=source,
        document_profile=_profile(width=64, height=32),
        canvas_width_m=20,
        archive_consent=False,
        preparation_seconds=0.02,
    )


def test_quote_uses_document_only_and_provider_checkout(monkeypatch):
    monkeypatch.setenv("ASTRA_TEST_USD_BRL", "5.50")
    monkeypatch.setenv("ASTRA_TEST_RETRY_RESERVE", "1.50")
    monkeypatch.setenv("ASTRA_TEST_FIXED_COST_BRL", "2.50")
    monkeypatch.setenv("ASTRA_TEST_PRICE_MULTIPLIER", "3.0")
    monkeypatch.setenv("PLAN_BIM_MIN_PRICE_BRL", "49.90")
    quote = quote_for_document(_profile(), _inspection())
    assert quote["currency"] == "BRL"
    assert quote["customer_price"] == 49.90
    assert quote["price_per_page"] == 49.90
    assert quote["pricing_version"] == "launch-fixed-area-reference-v1"
    assert quote["pricing_mode"] == "launch-fixed"
    assert quote["estimated_area_m2"] == 200
    assert quote["price_per_m2"] == 0.5
    assert quote["calculated_price_brl"] == 100
    assert quote["payment_mode"] == "mercado-pago"
    assert quote["basis"]["pricing_input"] == "local-area-reference-fixed-launch-price"
    assert quote["basis"]["heuristic_detector_used"] is False
    assert "candidate_count" not in quote["basis"]


def test_quote_is_fixed_for_one_page_regardless_of_area_or_image(monkeypatch):
    monkeypatch.setenv("ASTRA_TEST_USD_BRL", "5.50")
    monkeypatch.setenv("ASTRA_TEST_RETRY_RESERVE", "1.50")
    monkeypatch.setenv("ASTRA_TEST_FIXED_COST_BRL", "2.50")
    monkeypatch.setenv("ASTRA_TEST_PRICE_MULTIPLIER", "3.0")
    monkeypatch.setenv("PLAN_BIM_MIN_PRICE_BRL", "49.90")
    small = quote_for_document(_profile(), _inspection(200))
    same_area = quote_for_document(_profile(width=6000, height=3000), _inspection(200))
    large = quote_for_document(_profile(), _inspection(1000))
    assert same_area["customer_price"] == small["customer_price"]
    assert large["customer_price"] == small["customer_price"]


def test_unconfirmed_job_refreshes_a_legacy_quote(monkeypatch, tmp_path):
    monkeypatch.setenv("PLAN_BIM_MIN_PRICE_BRL", "49.90")
    manager = AstraLocalFlowManager(tmp_path / "jobs", enabled=True, max_workers=1)
    prepared = _create_job(manager, tmp_path)
    state = manager._read("0123456789")
    state["processing_mode"] = "legacy-hybrid"
    state["status"] = "awaiting_confirmation"
    state["quote"] = {"customer_price": 12.90, "payment_mode": "local-simulation"}
    manager._write(state)
    manager.close()

    manager = AstraLocalFlowManager(tmp_path / "jobs", enabled=True, max_workers=1)
    refreshed = manager.public_status("0123456789", prepared["access_token"])
    migrated = manager._read("0123456789")
    manager.close()
    assert migrated["processing_mode"] == "astra-direct"
    assert migrated["quote"]["pricing_version"] == "launch-fixed-area-reference-v1"
    assert refreshed["quote"]["customer_price"] >= 49.90
    assert refreshed["preanalysis"]["prepared"] is True
    assert "pricing_version" not in refreshed["quote"]
    assert "api_cost_estimate_usd" not in refreshed["quote"]


def test_local_flow_waits_for_verified_payment_then_processes_without_detector(
    monkeypatch, tmp_path
):
    from plantatobim import pre_wall_opening_import

    gateway = FakePaymentGateway()
    manager = AstraLocalFlowManager(
        tmp_path / "jobs",
        enabled=True,
        stage_factory=FakeStage,
        payment_gateway=gateway,
        max_workers=1,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(free_app, "ASTRA_FLOW_MANAGER", manager)

    def detector_must_not_run(*_args, **_kwargs):
        raise AssertionError("O detector heurístico entrou no fluxo Astra")

    monkeypatch.setattr(
        pre_wall_opening_import,
        "pre_wall_image_to_editor_model",
        detector_must_not_run,
    )

    client = free_app.app.test_client()
    preflight = client.post(
        "/api/astra-flow/preflight",
        data={
            "file": (io.BytesIO(_png_bytes()), "sample.png"),
            "canvas_width_m": "20",
            "archive_consent": "false",
        },
        content_type="multipart/form-data",
    )
    assert preflight.status_code == 201
    prepared = preflight.get_json()
    assert prepared["status"] == "awaiting_payment"
    assert prepared["quote"]["payment_mode"] == "mercado-pago"
    assert prepared["quote"]["checkout_available"] is True
    assert prepared["preanalysis"]["prepared"] is True
    assert "walls" not in prepared["preanalysis"]
    assert "api_cost_estimate_usd" not in prepared["quote"]
    assert "basis" not in prepared["quote"]
    assert prepared["ifc_generated"] is False
    job_dir = manager.job_dir(prepared["job"])
    internal_state = manager._read(prepared["job"])
    assert internal_state["processing_mode"] == "astra-direct"
    assert internal_state["preanalysis"]["geometry_detector_used"] is False
    assert not (job_dir / "detector_model.json").exists()
    assert not (job_dir / "detector").exists()

    denied = client.get(
        f"/api/astra-flow/jobs/{prepared['job']}?token=wrong-token"
    )
    assert denied.status_code == 404

    removed_confirmation = client.post(
        f"/api/astra-flow/jobs/{prepared['job']}/confirm",
        json={"access_token": prepared["access_token"], "payment_confirmed": True},
    )
    assert removed_confirmation.status_code == 404

    checkout = client.post(
        f"/api/astra-flow/jobs/{prepared['job']}/checkout",
        json={"access_token": prepared["access_token"]},
    )
    assert checkout.status_code == 201
    checkout_payload = checkout.get_json()
    assert checkout_payload["status"] == "awaiting_payment"
    assert checkout_payload["checkout_url"].startswith("https://sandbox.mercadopago.com.br/")
    assert len(gateway.preferences) == 1
    assert manager._read(prepared["job"])["status"] == "awaiting_payment"
    assert "preference_id" not in checkout_payload
    assert "provider_payment_id" not in checkout_payload

    repeated_checkout = client.post(
        f"/api/astra-flow/jobs/{prepared['job']}/checkout",
        json={"access_token": prepared["access_token"]},
    )
    assert repeated_checkout.status_code == 201
    assert len(gateway.preferences) == 1

    gateway.payments = [_approved_payment(manager, prepared["job"])]
    synchronization = client.post(
        f"/api/astra-flow/jobs/{prepared['job']}/payment/sync",
        json={"access_token": prepared["access_token"]},
    )
    assert synchronization.status_code == 200
    assert synchronization.get_json()["status"] in {"queued", "running", "completed"}

    deadline = time.monotonic() + 3
    status = None
    while time.monotonic() < deadline:
        response = client.get(
            f"/api/astra-flow/jobs/{prepared['job']}"
            f"?token={prepared['access_token']}"
        )
        status = response.get_json()
        if status["status"] == "completed":
            break
        time.sleep(0.02)

    manager.close()
    assert status is not None
    assert status["status"] == "completed"
    assert status["progress"] == 100
    assert status["stage"] == "editor_ready"
    assert status["result"]["paredes"][0]["nome"] == "Parede Astra"
    assert status["result"]["astra_editor"]["status"] == "ready-for-manual-review"
    assert status["result"]["engine"] == "pro-analysis-v1"
    assert status["result"]["source"]["mode"] == "pro-analysis"
    assert "heuristic_detector_used" not in status["result"]["source"]
    assert "astra_direct" not in status["result"]
    assert status["analysis"]["needs_human_review"] is False
    assert "semantic" not in status
    assert "api" not in status
    assert status["ifc_generated"] is False
    assert not list(tmp_path.rglob("*.ifc"))


def test_payment_amount_mismatch_never_starts_processing(monkeypatch, tmp_path):
    calls = {"count": 0}

    class CountingStage(FakeStage):
        def analyze(self, *args, **kwargs):
            calls["count"] += 1
            return super().analyze(*args, **kwargs)

    gateway = FakePaymentGateway()
    manager = AstraLocalFlowManager(
        tmp_path / "jobs",
        enabled=True,
        stage_factory=CountingStage,
        payment_gateway=gateway,
        max_workers=1,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    prepared = _create_job(manager, tmp_path)
    manager.create_checkout("0123456789", prepared["access_token"])
    payment = _approved_payment(manager, "0123456789")
    payment["transaction_amount"] = 0.01
    gateway.payments = [payment]

    status = manager.sync_payment("0123456789", prepared["access_token"])
    internal = manager._read("0123456789")
    manager.close()

    assert status["status"] == "awaiting_payment"
    assert status["payment"]["status"] == "review_required"
    assert internal["stage"] == "payment_review_required"
    assert calls["count"] == 0


def test_failed_call_requires_explicit_retry(monkeypatch, tmp_path):
    attempts = {"count": 0}

    class FlakyStage:
        def analyze(self, image_path, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary failure")
            return FakeStage().analyze(image_path, **kwargs)

    gateway = FakePaymentGateway()
    manager = AstraLocalFlowManager(
        tmp_path / "jobs",
        enabled=True,
        stage_factory=FlakyStage,
        payment_gateway=gateway,
        max_workers=1,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    prepared = _create_job(manager, tmp_path)
    token = prepared["access_token"]
    manager.create_checkout("0123456789", token)
    gateway.payments = [_approved_payment(manager, "0123456789")]
    manager.sync_payment("0123456789", token)

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        failed = manager.public_status("0123456789", token)
        if failed["status"] == "failed":
            break
        time.sleep(0.02)
    assert failed["status"] == "failed"
    manager.retry("0123456789", token)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        completed = manager.public_status("0123456789", token)
        if completed["status"] == "completed":
            break
        time.sleep(0.02)
    manager.close()
    assert completed["status"] == "completed"
    assert completed["retry_count"] == 1


def test_restart_marks_in_process_job_as_interrupted(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    root = tmp_path / "jobs"
    first = AstraLocalFlowManager(root, enabled=True, max_workers=1)
    prepared = _create_job(first, tmp_path)
    state_path = root / "0123456789" / "job.json"
    state = first._read("0123456789")
    state["status"] = "running"
    state["stage"] = "astra_direct_analysis"
    first._write(state)
    first.close()

    recovered = AstraLocalFlowManager(root, enabled=True, max_workers=1)
    status = recovered.public_status("0123456789", prepared["access_token"])
    recovered.close()
    assert state_path.is_file()
    assert status["status"] == "failed"
    assert status["stage"] == "backend_restarted"
    assert status["retry_requires_confirmation"] is True


def test_durable_flow_queues_paid_job_and_reads_result_from_storage(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    gateway = FakePaymentGateway()
    store = FakeDurableStore()
    dispatcher = FakeDispatcher()
    manager = AstraLocalFlowManager(
        tmp_path / "jobs",
        enabled=True,
        stage_factory=FakeStage,
        payment_gateway=gateway,
        job_store=store,
        dispatcher=dispatcher,
    )
    prepared = _create_job(manager, tmp_path)
    token = prepared["access_token"]
    manager.create_checkout("0123456789", token)
    gateway.payments = [_approved_payment(manager, "0123456789")]
    queued = manager.sync_payment("0123456789", token)

    assert queued["status"] == "queued"
    assert len(dispatcher.tasks) == 1
    assert dispatcher.tasks[0][:2] == ("0123456789", 0)
    assert "queue" not in queued

    completed = manager.run_worker("0123456789")
    assert completed["status"] == "completed"
    Path(completed["result_path"]).unlink()
    public = manager.public_status("0123456789", token)
    manager.close()

    assert public["result"]["paredes"][0]["nome"] == "Parede Astra"
    assert "result_object" not in public
    assert "analysis_object" not in public


def test_durable_restart_does_not_fail_cloud_tasks_job(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    store = FakeDurableStore()
    dispatcher = FakeDispatcher()
    manager = AstraLocalFlowManager(
        tmp_path / "jobs",
        enabled=True,
        job_store=store,
        dispatcher=dispatcher,
    )
    prepared = _create_job(manager, tmp_path)
    state = manager._read("0123456789")
    state["status"] = "running"
    state["payment"] = {"status": "approved", "confirmed": True}
    manager._write(state)
    manager.close()

    restarted = AstraLocalFlowManager(
        tmp_path / "jobs",
        enabled=True,
        job_store=store,
        dispatcher=dispatcher,
    )
    status = restarted.public_status("0123456789", prepared["access_token"])
    restarted.close()
    assert status["status"] == "running"


def test_second_paid_job_waits_until_same_owner_finishes(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    gateway = FakePaymentGateway()
    store = FakeDurableStore()
    dispatcher = FakeDispatcher()
    manager = AstraLocalFlowManager(
        tmp_path / "jobs",
        enabled=True,
        stage_factory=FakeStage,
        payment_gateway=gateway,
        job_store=store,
        dispatcher=dispatcher,
    )
    source = tmp_path / "sample.png"
    source.write_bytes(_png_bytes())

    first = manager.create_job(
        job="0123456789",
        original_name="sample.png",
        original_path=source,
        image_path=source,
        document_profile=_profile(width=64, height=32),
        canvas_width_m=20,
        archive_consent=False,
        preparation_seconds=0.01,
        owner_id="owner-1",
    )
    second = manager.create_job(
        job="abcdef1234",
        original_name="sample.png",
        original_path=source,
        image_path=source,
        document_profile=_profile(width=64, height=32),
        canvas_width_m=20,
        archive_consent=False,
        preparation_seconds=0.01,
        owner_id="owner-1",
    )
    manager.create_checkout("0123456789", first["access_token"])
    manager.create_checkout("abcdef1234", second["access_token"])

    first_payment = _approved_payment(manager, "0123456789")
    second_payment = _approved_payment(manager, "abcdef1234")
    gateway.payments = [first_payment, second_payment]
    manager.sync_payment("0123456789", first["access_token"])
    waiting = manager.sync_payment("abcdef1234", second["access_token"])

    assert waiting["status"] == "waiting_for_slot"
    assert len(dispatcher.tasks) == 1

    manager.run_worker("0123456789")
    promoted = manager.public_status("abcdef1234", second["access_token"])
    manager.close()

    assert promoted["status"] == "queued"
    assert len(dispatcher.tasks) == 2


def test_simultaneous_payment_notifications_enqueue_only_one_owner_job(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    gateway = FakePaymentGateway()
    store = FakeDurableStore()
    dispatcher = FakeDispatcher()
    manager = AstraLocalFlowManager(
        tmp_path / "jobs",
        enabled=True,
        stage_factory=FakeStage,
        payment_gateway=gateway,
        job_store=store,
        dispatcher=dispatcher,
    )
    source = tmp_path / "sample.png"
    source.write_bytes(_png_bytes())
    jobs = []
    for job in ("0123456789", "abcdef1234"):
        created = manager.create_job(
            job=job,
            original_name="sample.png",
            original_path=source,
            image_path=source,
            document_profile=_profile(width=64, height=32),
            canvas_width_m=20,
            archive_consent=False,
            preparation_seconds=0.01,
            owner_id="owner-concurrent",
        )
        manager.create_checkout(job, created["access_token"])
        jobs.append((job, created["access_token"]))

    payments = []
    for index, (job, _token) in enumerate(jobs, start=1):
        payment = _approved_payment(manager, job)
        payment["id"] = str(index)
        payments.append(payment)
    gateway.payments = payments

    barrier = threading.Barrier(2)
    original_get_payment = gateway.get_payment

    def synchronized_get_payment(payment_id):
        barrier.wait(timeout=2)
        return original_get_payment(payment_id)

    gateway.get_payment = synchronized_get_payment
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(manager.handle_payment_notification, ("1", "2")))

    statuses = {manager._read(job)["status"] for job, _token in jobs}
    manager.close()
    assert results == [True, True]
    assert statuses == {"queued", "waiting_for_slot"}
    assert len(dispatcher.tasks) == 1


def test_webhook_rejects_invalid_signature_and_replay_is_idempotent(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    gateway = FakePaymentGateway()
    store = FakeDurableStore()
    dispatcher = FakeDispatcher()
    manager = AstraLocalFlowManager(
        tmp_path / "jobs",
        enabled=True,
        stage_factory=FakeStage,
        payment_gateway=gateway,
        job_store=store,
        dispatcher=dispatcher,
    )
    prepared = _create_job(manager, tmp_path)
    manager.create_checkout("0123456789", prepared["access_token"])
    gateway.payments = [_approved_payment(manager, "0123456789")]
    gateway.verify_webhook = lambda **headers: headers.get("x_signature") == "valid"
    monkeypatch.setattr(free_app, "ASTRA_FLOW_MANAGER", manager)
    monkeypatch.setattr(free_app, "ASSISTED_ORDER_ENABLED", False)
    client = free_app.app.test_client()
    payload = {"type": "payment", "data": {"id": "987654321"}}

    invalid = client.post(
        "/api/payments/mercadopago/webhook",
        json=payload,
        headers={"x-signature": "invalid", "x-request-id": "request-1"},
    )
    assert invalid.status_code == 401
    assert dispatcher.tasks == []

    first = client.post(
        "/api/payments/mercadopago/webhook",
        json=payload,
        headers={"x-signature": "valid", "x-request-id": "request-1"},
    )
    replay = client.post(
        "/api/payments/mercadopago/webhook",
        json=payload,
        headers={"x-signature": "valid", "x-request-id": "request-1"},
    )
    manager.close()

    assert first.status_code == 200
    assert replay.status_code == 200
    assert len(dispatcher.tasks) == 1


def test_internal_worker_rejects_public_request(monkeypatch):
    class RejectingManager:
        def verify_worker_request(self, authorization):
            raise PermissionError("invalid")

    monkeypatch.setattr(free_app, "ASTRA_FLOW_MANAGER", RejectingManager())
    response = free_app.app.test_client().post(
        "/api/internal/astra-flow/jobs/0123456789/run",
        json={"job": "0123456789"},
    )
    assert response.status_code == 401


def test_internal_worker_runs_only_after_identity_verification(monkeypatch):
    calls = []

    class VerifiedManager:
        def verify_worker_request(self, authorization):
            calls.append(("verify", authorization))

        def run_worker(self, job):
            calls.append(("run", job))
            return {"job": job, "status": "completed"}

    monkeypatch.setattr(free_app, "ASTRA_FLOW_MANAGER", VerifiedManager())
    response = free_app.app.test_client().post(
        "/api/internal/astra-flow/jobs/0123456789/run",
        headers={"Authorization": "Bearer task-token"},
        json={"job": "0123456789"},
    )
    assert response.status_code == 200
    assert response.get_json()["status"] == "completed"
    assert calls == [
        ("verify", "Bearer task-token"),
        ("run", "0123456789"),
    ]
