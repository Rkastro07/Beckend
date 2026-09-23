from pathlib import Path


def test_first_preview_scopes_are_allowed_by_table_and_rpc():
    migration = (
        Path(__file__).resolve().parents[1]
        / "supabase/migrations/202609230002_plan_bim_first_preview_rate_limits.sql"
    ).read_text(encoding="utf-8")
    assert "drop constraint if exists plan_bim_rate_limits_scope_check" in migration
    assert "create or replace function public.consume_plan_bim_rate_limit" in migration
    for scope in ("first-preview-global", "first-preview-ip"):
        assert migration.count(f"'{scope}'") == 2
