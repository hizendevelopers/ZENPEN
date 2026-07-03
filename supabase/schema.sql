-- Supabase schema for Media Product Suite
-- Run this in the Supabase SQL editor.

create extension if not exists "pgcrypto";

-- Profiles extend Supabase Auth users
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  full_name text,
  avatar_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, full_name, avatar_url)
  values (
    new.id,
    new.raw_user_meta_data ->> 'full_name',
    new.raw_user_meta_data ->> 'avatar_url'
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute procedure public.handle_new_user();

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists public.products (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique,
  title text not null,
  description text,
  banner_url text,
  status text not null default 'coming_soon' check (status in ('ready', 'coming_soon')),
  created_at timestamptz not null default now()
);

insert into public.products (slug, title, description, status)
values
  ('article-generator', 'Article Generator', 'Analyze media and generate articles from detected topics.', 'ready'),
  ('blog-generator', 'Blog Generator', 'Generate blog-focused content workflows.', 'coming_soon'),
  ('srt-file-generator', '.SRT File Generator', 'Create subtitle and caption files from media.', 'coming_soon')
on conflict (slug) do update
set
  title = excluded.title,
  description = excluded.description,
  status = excluded.status;

create table if not exists public.analysis_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  product_id uuid not null references public.products(id) on delete restrict,
  source_type text not null check (source_type in ('url', 'upload')),
  source_url text,
  source_file_name text,
  source_file_path text,
  source_mime_type text,
  query text not null default 'Give breaking news and main points',
  status text not null default 'completed' check (status in ('queued', 'processing', 'completed', 'failed')),
  headline text,
  summary text,
  error_message text,
  raw_result jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  completed_at timestamptz
);

create trigger set_analysis_jobs_updated_at
before update on public.analysis_jobs
for each row execute procedure public.set_updated_at();

create table if not exists public.analysis_topics (
  id uuid primary key default gen_random_uuid(),
  analysis_job_id uuid not null references public.analysis_jobs(id) on delete cascade,
  topic text not null,
  sort_order integer not null default 0,
  selected boolean not null default false,
  created_at timestamptz not null default now()
);

create unique index if not exists analysis_topics_job_topic_idx
  on public.analysis_topics (analysis_job_id, topic);

create table if not exists public.generated_articles (
  id uuid primary key default gen_random_uuid(),
  analysis_job_id uuid not null references public.analysis_jobs(id) on delete cascade,
  topic_id uuid references public.analysis_topics(id) on delete set null,
  title text not null,
  topic text not null,
  content text not null,
  image_url text,
  sort_order integer not null default 0,
  created_at timestamptz not null default now()
);

create table if not exists public.user_product_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  product_id uuid not null references public.products(id) on delete cascade,
  event_type text not null check (event_type in ('viewed', 'started', 'completed')),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists analysis_jobs_user_created_idx
  on public.analysis_jobs (user_id, created_at desc);

create index if not exists generated_articles_job_created_idx
  on public.generated_articles (analysis_job_id, created_at desc);

create index if not exists user_product_events_user_created_idx
  on public.user_product_events (user_id, created_at desc);

alter table public.profiles enable row level security;
alter table public.products enable row level security;
alter table public.analysis_jobs enable row level security;
alter table public.analysis_topics enable row level security;
alter table public.generated_articles enable row level security;
alter table public.user_product_events enable row level security;

-- Profiles
drop policy if exists "Users can read own profile" on public.profiles;
create policy "Users can read own profile"
on public.profiles
for select
to authenticated
using (auth.uid() = id);

drop policy if exists "Users can update own profile" on public.profiles;
create policy "Users can update own profile"
on public.profiles
for update
to authenticated
using (auth.uid() = id)
with check (auth.uid() = id);

-- Products are public-read
drop policy if exists "Anyone can read products" on public.products;
create policy "Anyone can read products"
on public.products
for select
to anon, authenticated
using (true);

-- Analysis jobs
drop policy if exists "Users can read own analysis jobs" on public.analysis_jobs;
create policy "Users can read own analysis jobs"
on public.analysis_jobs
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "Users can insert own analysis jobs" on public.analysis_jobs;
create policy "Users can insert own analysis jobs"
on public.analysis_jobs
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "Users can update own analysis jobs" on public.analysis_jobs;
create policy "Users can update own analysis jobs"
on public.analysis_jobs
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "Users can delete own analysis jobs" on public.analysis_jobs;
create policy "Users can delete own analysis jobs"
on public.analysis_jobs
for delete
to authenticated
using (auth.uid() = user_id);

-- Topics
drop policy if exists "Users can read own topics" on public.analysis_topics;
create policy "Users can read own topics"
on public.analysis_topics
for select
to authenticated
using (
  exists (
    select 1
    from public.analysis_jobs aj
    where aj.id = analysis_job_id
      and aj.user_id = auth.uid()
  )
);

drop policy if exists "Users can insert own topics" on public.analysis_topics;
create policy "Users can insert own topics"
on public.analysis_topics
for insert
to authenticated
with check (
  exists (
    select 1
    from public.analysis_jobs aj
    where aj.id = analysis_job_id
      and aj.user_id = auth.uid()
  )
);

drop policy if exists "Users can update own topics" on public.analysis_topics;
create policy "Users can update own topics"
on public.analysis_topics
for update
to authenticated
using (
  exists (
    select 1
    from public.analysis_jobs aj
    where aj.id = analysis_job_id
      and aj.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1
    from public.analysis_jobs aj
    where aj.id = analysis_job_id
      and aj.user_id = auth.uid()
  )
);

-- Articles
drop policy if exists "Users can read own articles" on public.generated_articles;
create policy "Users can read own articles"
on public.generated_articles
for select
to authenticated
using (
  exists (
    select 1
    from public.analysis_jobs aj
    where aj.id = analysis_job_id
      and aj.user_id = auth.uid()
  )
);

drop policy if exists "Users can insert own articles" on public.generated_articles;
create policy "Users can insert own articles"
on public.generated_articles
for insert
to authenticated
with check (
  exists (
    select 1
    from public.analysis_jobs aj
    where aj.id = analysis_job_id
      and aj.user_id = auth.uid()
  )
);

drop policy if exists "Users can update own articles" on public.generated_articles;
create policy "Users can update own articles"
on public.generated_articles
for update
to authenticated
using (
  exists (
    select 1
    from public.analysis_jobs aj
    where aj.id = analysis_job_id
      and aj.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1
    from public.analysis_jobs aj
    where aj.id = analysis_job_id
      and aj.user_id = auth.uid()
  )
);

drop policy if exists "Users can delete own articles" on public.generated_articles;
create policy "Users can delete own articles"
on public.generated_articles
for delete
to authenticated
using (
  exists (
    select 1
    from public.analysis_jobs aj
    where aj.id = analysis_job_id
      and aj.user_id = auth.uid()
  )
);

-- Events
drop policy if exists "Users can manage own product events" on public.user_product_events;
create policy "Users can manage own product events"
on public.user_product_events
for all
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);
