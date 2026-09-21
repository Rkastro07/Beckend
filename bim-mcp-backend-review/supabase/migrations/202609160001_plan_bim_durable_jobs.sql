-- Durable state for paid Plan2BIM conversions. The browser never reads this
-- table directly; only the Cloud Run service role has access.

create table if not exists public.plan_bim_jobs (
  job_id text primary key check (job_id ~ '^[a-f0-9]{10}$'),
  owner_id text not null default '',
  owner_email text not null default '',
  status text not null,
  external_reference text,
  state jsonb not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists plan_bim_jobs_external_reference_uidx
  on public.plan_bim_jobs (external_reference)
  where external_reference is not null;

create index if not exists plan_bim_jobs_owner_updated_idx
  on public.plan_bim_jobs (owner_id, updated_at desc);

create index if not exists plan_bim_jobs_status_updated_idx
  on public.plan_bim_jobs (status, updated_at desc);

alter table public.plan_bim_jobs enable row level security;
revoke all on table public.plan_bim_jobs from anon, authenticated;
grant select, insert, update, delete on table public.plan_bim_jobs to service_role;

insert into storage.buckets (id, name, public, file_size_limit)
values ('plan2bim-jobs', 'plan2bim-jobs', false, 31457280)
on conflict (id) do update
set public = false,
    file_size_limit = excluded.file_size_limit;
