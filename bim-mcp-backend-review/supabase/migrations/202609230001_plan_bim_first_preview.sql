-- One introductory analysis per account. Only the server's service role can
-- reserve a preview. Existing accounts with prior conversion jobs are not
-- eligible; a claim is permanent even if the browser closes.
create table if not exists public.plan_bim_first_previews (
  owner_id text primary key,
  job_id text not null unique check (job_id ~ '^[a-f0-9]{10}$'),
  claimed_at timestamptz not null default now()
);

alter table public.plan_bim_first_previews enable row level security;
revoke all on public.plan_bim_first_previews from public, anon, authenticated;
grant select, insert on public.plan_bim_first_previews to service_role;

create or replace function public.claim_plan_bim_first_preview(
  p_owner_id text, p_job_id text
) returns boolean
language plpgsql security definer set search_path = public
as $$
begin
  if nullif(trim(p_owner_id), '') is null
     or p_job_id !~ '^[a-f0-9]{10}$' then
    return false;
  end if;
  insert into public.plan_bim_first_previews (owner_id, job_id)
  select p_owner_id, p_job_id
  where not exists (
    select 1 from public.plan_bim_jobs where owner_id = p_owner_id
  )
  on conflict do nothing;
  return found;
end;
$$;

revoke all on function public.claim_plan_bim_first_preview(text, text) from public, anon, authenticated;
grant execute on function public.claim_plan_bim_first_preview(text, text) to service_role;
