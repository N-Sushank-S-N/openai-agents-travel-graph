-- Run this once in Supabase SQL Editor.
-- Prefer SUPABASE_SERVICE_ROLE_KEY as a server-side Render secret; never expose it in browser code.
create table if not exists public.travel_plan_runs (
    session_key text primary key,
    created_at timestamptz not null default now(),
    origin text,
    destination text,
    request jsonb,
    preferences jsonb,
    plan jsonb
);

create index if not exists travel_plan_runs_created_at_idx
    on public.travel_plan_runs (created_at desc);

alter table public.travel_plan_runs enable row level security;
-- No anon policies are created intentionally. Server-side service-role access bypasses RLS.
