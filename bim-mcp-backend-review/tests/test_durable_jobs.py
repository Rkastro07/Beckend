from __future__ import annotations

from plantatobim.durable_jobs import SupabaseJobStore


def test_retention_filter_percent_encodes_positive_utc_offset(monkeypatch):
    store = SupabaseJobStore(
        project_url="https://example.supabase.co",
        secret_key="service-key",
    )
    endpoints = []

    def fake_call(method, endpoint, **kwargs):
        endpoints.append((method, endpoint, kwargs))
        return b"[]"

    monkeypatch.setattr(store, "_call", fake_call)
    assert store.purge_expired_files(retention_days=30) == {
        "purged_jobs": 0,
        "skipped_jobs": 0,
    }
    assert endpoints[0][0] == "GET"
    assert "%2B00:00" in endpoints[0][1]
    assert "+00:00" not in endpoints[0][1]
