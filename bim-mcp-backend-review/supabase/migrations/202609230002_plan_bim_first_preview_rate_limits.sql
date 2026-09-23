-- Extend the durable fixed-window limiter for the first-preview flow.
-- This is separate from the preview claim migration so it also upgrades
-- projects that already applied the original rate-limit migration.

alter table public.plan_bim_rate_limits
  drop constraint if exists plan_bim_rate_limits_scope_check;
alter table public.plan_bim_rate_limits
  add constraint plan_bim_rate_limits_scope_check check (scope in (
    'upload-user', 'upload-ip', 'order-user', 'order-ip',
    'first-preview-global', 'first-preview-ip'
  ));

create or replace function public.consume_plan_bim_rate_limit(
  p_scope text,
  p_identity_hash text,
  p_limit integer,
  p_window_seconds integer
)
returns table (
  allowed boolean,
  retry_after_seconds integer,
  current_count integer
)
language plpgsql
security definer
set search_path = public
as $$
declare
  v_now timestamptz := clock_timestamp();
  v_window_start timestamptz;
  v_count integer;
begin
  if p_scope not in (
    'upload-user', 'upload-ip', 'order-user', 'order-ip',
    'first-preview-global', 'first-preview-ip'
  ) then
    raise exception 'invalid rate-limit scope';
  end if;
  if p_identity_hash !~ '^[a-f0-9]{64}$' then
    raise exception 'invalid identity hash';
  end if;
  if p_limit < 1 or p_window_seconds < 1 then
    raise exception 'invalid rate-limit configuration';
  end if;

  v_window_start := to_timestamp(
    floor(extract(epoch from v_now) / p_window_seconds) * p_window_seconds
  );

  insert into public.plan_bim_rate_limits (
    scope, identity_hash, window_started_at, window_seconds, event_count, updated_at
  ) values (
    p_scope, p_identity_hash, v_window_start, p_window_seconds, 0, v_now
  )
  on conflict (scope, identity_hash, window_started_at) do nothing;

  select event_count
    into v_count
    from public.plan_bim_rate_limits
   where scope = p_scope
     and identity_hash = p_identity_hash
     and window_started_at = v_window_start
   for update;

  if v_count >= p_limit then
    return query select
      false,
      greatest(1, ceil(extract(epoch from (
        v_window_start + make_interval(secs => p_window_seconds) - v_now
      )))::integer),
      v_count;
    return;
  end if;

  update public.plan_bim_rate_limits
     set event_count = event_count + 1,
         updated_at = v_now
   where scope = p_scope
     and identity_hash = p_identity_hash
     and window_started_at = v_window_start
  returning event_count into v_count;

  return query select
    true,
    greatest(1, ceil(extract(epoch from (
      v_window_start + make_interval(secs => p_window_seconds) - v_now
    )))::integer),
    v_count;
end;
$$;

revoke all on function public.consume_plan_bim_rate_limit(text, text, integer, integer)
  from public, anon, authenticated;
grant execute on function public.consume_plan_bim_rate_limit(text, text, integer, integer)
  to service_role;
