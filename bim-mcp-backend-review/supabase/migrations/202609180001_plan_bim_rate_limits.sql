-- Global fixed-window limits for Plan2BIM public operations. Only the backend
-- service role can execute the function; raw user IDs and IP addresses are
-- HMAC-fingerprinted before they reach Postgres.

create table if not exists public.plan_bim_rate_limits (
  scope text not null check (scope in (
    'upload-user', 'upload-ip', 'order-user', 'order-ip'
  )),
  identity_hash text not null check (identity_hash ~ '^[a-f0-9]{64}$'),
  window_started_at timestamptz not null,
  window_seconds integer not null check (window_seconds > 0),
  event_count integer not null default 0 check (event_count >= 0),
  updated_at timestamptz not null default now(),
  primary key (scope, identity_hash, window_started_at)
);

create index if not exists plan_bim_rate_limits_updated_idx
  on public.plan_bim_rate_limits (updated_at);

alter table public.plan_bim_rate_limits enable row level security;
revoke all on table public.plan_bim_rate_limits from anon, authenticated;
grant select, insert, update, delete on table public.plan_bim_rate_limits to service_role;

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
  if p_scope not in ('upload-user', 'upload-ip', 'order-user', 'order-ip') then
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
