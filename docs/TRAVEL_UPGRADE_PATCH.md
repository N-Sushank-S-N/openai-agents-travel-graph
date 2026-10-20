# Travel Planner: architecture completion patch

This patch closes several implementation gaps in the deployed travel planner while preserving the existing Streamlit UI and LangGraph workflow.

## What this patch adds

1. **OpenAI Agents SDK at runtime**: replaces the query-analysis node's orchestrator with a drop-in adapter that actually calls `Agent` and `Runner`. Five specialist Agents SDK agents are delegated as tools: destination research, transport, accommodation, activities/itinerary, and budget. The existing LangGraph workflow still controls the overall workflow stages.
2. **Gemini compatible**: uses `GEMINI_API_KEY` + the Gemini OpenAI-compatible endpoint by default when a Gemini key exists. It does not require an OpenAI API key or OpenAI usage fees. Set `TRAVEL_LLM_PROVIDER=openai` only if you intentionally want to use OpenAI.
3. **Optional Playwright browser research**: agents can extract rendered text from public pages. The tool blocks non-HTTP(S), localhost, internal hosts, and non-public IP addresses; it does not log in, book, or submit forms. Per-run browser calls are capped.
4. **Supabase completed-plan persistence**: the final LangGraph node is wrapped to upsert each completed plan into `travel_plan_runs`, if Supabase env variables are configured. A failed save does not break the itinerary.
5. **SQL schema**: `supabase/travel_plan_runs.sql` creates the dedicated persistence table.

## Apply locally

Copy this installer into the repository root (the folder containing `pyproject.toml`) and run:

```powershell
python .\apply_travel_upgrade.py --install-sdk
```

The installer backs up files it changes, patches the Streamlit preference form, and uses `uv add openai-agents` to update both `pyproject.toml` and `uv.lock`. It does not install a browser binary during the patch step.

The script backs up modified files before changing them. It is designed to be re-run safely.

## Environment variables

- `TRAVEL_USE_OPENAI_AGENTS_SDK=1` (default) uses the SDK adapter.
- `TRAVEL_LLM_PROVIDER=gemini` (recommended for this project) or `openai`.
- Gemini: `GEMINI_API_KEY`, optional `GEMINI_BASE_URL`, `GEMINI_MODEL`.
- OpenAI: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_MODEL`.
- Optional custom compatible provider: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
- Supabase: `SUPABASE_URL` and preferably server-side `SUPABASE_SERVICE_ROLE_KEY`. Never put this key in browser code or commit `.env`.
- `TRAVEL_BROWSER_CALL_LIMIT=2` (capped to 3).

For Supabase persistence, run `supabase/travel_plan_runs.sql` in the Supabase SQL Editor. Keep the existing `SUPABASE_URL` and `SUPABASE_KEY` values required by the current app. For the new table, configure `SUPABASE_SERVICE_ROLE_KEY` in Render (server side only) unless you have confirmed the existing `SUPABASE_KEY` is a service-role/write-capable key. No public anonymous database write policy is created.

## Playwright

The project already lists Playwright. Install Chromium in a development environment with:

```powershell
uv run playwright install chromium
```

The current Render Blueprint uses the native Python runtime and does not install Chromium automatically. Browser tool code is integrated, but it will report that Playwright cannot start until Chromium and its OS libraries are available. On Windows run `uv run playwright install chromium`; for the current Render service, enable browser execution only after installing a compatible Chromium runtime (a Docker image based on the official Playwright image is the most reliable option, but would require a separate deployment change).

## Validate

```powershell
uv sync --extra dev --locked
python -m compileall travel_planner
python -m pytest tests\unit -q
```

The Streamlit form also collects travel style, interests, accommodation preference, trip pace, and special requirements; those values are passed into the workflow query. Then run one itinerary in the deployed app and check logs for SDK activity and `Travel plan saved to Supabase travel_plan_runs table`. Verify the row in Supabase Table Editor. This pack cannot push to GitHub or change Render settings for you; it prepares reproducible local changes only.
