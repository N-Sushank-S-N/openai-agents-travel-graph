# AI Travel Planner 🧭

An AI-assisted travel-planning MVP that combines destination research with a user-triggered price-comparison workspace. It generates itinerary guidance for a trip and can fetch provider-backed flight/hotel search snapshots and train timetable data when the relevant API keys are configured.

> **Important:** This application does not book or reserve tickets or rooms. Search results are snapshots, not a guarantee that a fare, room, or seat is still available. AI-generated cost breakdowns are estimates unless a figure is explicitly returned by a provider. Always confirm prices and availability with the booking provider before purchasing.

## Features

- **AI itinerary generation:** Enter an origin, destination, trip duration, traveler count, and budget to generate a day-by-day itinerary using Gemini.
- **Destination research:** Tavily search and Firecrawl page extraction add web-sourced context to the planning workflow. Research content is treated as reference material, not executable instructions.
- **Flight price snapshots:** Optional SerpApi Google Flights results when valid departure/arrival airport IATA codes and `SERPAPI_API_KEY` are supplied.
- **Hotel price snapshots:** Optional SerpApi Google Hotels results for selected dates and destination.
- **Train timetables:** Optional RailRadar trains-between-stations lookup. The current integration displays timetable results; it does **not** claim confirmed IRCTC seat availability or live ticket fares.
- **Bus search:** Opens a route/date search on an external booking website. Bus seat inventory is not automatically retrieved into the app.
- **Quota-conscious API calls:** Provider search calls are user-triggered, use a single attempt, and avoid automatic retries. Check the current provider quotas before use.
- **Checkpointing and workflow orchestration:** The CLI workflow uses a LangGraph-based orchestration path and can save workflow checkpoints.

## Current scope and limitations

This is an **itinerary-generation and travel-research MVP**, not a full online travel agency or booking engine.

| Data type | Current behaviour | What must be confirmed externally |
|---|---|---|
| Itinerary | Gemini-generated suggestions grounded with research context | Opening hours, route timings, and suitability for the travel date |
| Flight search | Optional SerpApi Google Flights snapshot | Final fare, baggage/rules, and bookability on the provider site |
| Hotel search | Optional SerpApi Google Hotels snapshot | Exact room price, taxes, cancellation terms, and availability |
| Trains | RailRadar timetable lookup when configured | Ticket fare and current seat availability on [IRCTC](https://www.irctc.co.in/nget/train-search) |
| Buses | External route-search link | Current operators, fare, departure time, and available seats on the booking site |
| Budget | AI-generated estimate, with any provider results shown separately | Final trip total based on confirmed prices |

Not every provider will return results for every origin/date. Vellore-to-Chennai is generally more suited to train and bus discovery than flights. Flight searches require valid airport IATA codes; do not substitute a city or railway-station code for an airport code.

## Tech stack

- Python 3.12
- Streamlit UI (`web_app.py`)
- Gemini API through Google's OpenAI-compatible endpoint
- LangGraph workflow orchestration
- Tavily search and Firecrawl extraction
- SerpApi for optional Google Flights/Hotels search snapshots
- RailRadar for optional train timetable lookup
- `uv` for dependency and lockfile management
- pytest for tests

## Project layout

```text
.
├── web_app.py                         # Streamlit user interface
├── render.yaml                        # Render deployment Blueprint
├── pyproject.toml                     # Project dependencies and configuration
├── uv.lock                            # Locked dependency versions
├── travel_planner/
│   ├── main.py                        # CLI entry point
│   ├── agents/                        # Specialized planning agents
│   ├── orchestration/                 # Workflow graph, nodes, state, checkpointing
│   └── utils/
│       ├── web_research_once.py       # Shared Tavily/Firecrawl research path
│       └── live_travel_provider.py    # SerpApi and RailRadar adapters
└── tests/
    └── unit/                          # Unit and mocked provider tests
```

Some filenames may differ if you have not copied every optional deployment or provider-integration file into your working tree. Use `git status --short` and the verification commands below to confirm your local files.

## Requirements

- Python 3.12.x
- [`uv`](https://docs.astral.sh/uv/)
- API keys for the features you plan to enable
- A Supabase project and valid credentials if required by the current application's configuration checks

## Local setup

### 1. Clone your repository and enter the folder

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create and activate a virtual environment (if needed)

```powershell
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
```

### 3. Install the project and development dependencies

```powershell
uv sync --extra dev --locked
```

Use `--extra dev` when running tests. A production deployment can install runtime dependencies without development tools. Do not use `uv sync --locked` alone if your workflow depends on packages that are only declared in the `dev` extra.

### 4. Configure environment variables

Create a local `.env` file from `.env.example` and fill only the keys you have legitimately obtained. See [API configuration](#api-configuration) below. Never commit `.env`.

### 5. Start the Streamlit app

```powershell
uv run streamlit run web_app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`.

### 6. Run the command-line planner

```powershell
python -m travel_planner.main --query "Plan a 2-day trip to Chennai" --origin "Vellore" --destination "Chennai" --budget "5000-8000"
```

For available command-line options:

```powershell
python -m travel_planner.main --help
```

### 7. Run tests

```powershell
python -m pytest tests\unit -q
```

If pytest reports that `pytest_asyncio` is missing, restore development dependencies:

```powershell
uv sync --extra dev --locked
```

Provider unit tests should use mocks and should not require live provider calls. Do not repeatedly run the full travel query to diagnose a unit-test failure, because it can consume API quota.

## API configuration

The following names are environment-variable names only. Put the **actual secret values in your local `.env` file or the hosting provider's secret settings**, never in this README or another committed file.

| Variable | Purpose | Required? |
|---|---|---|
| `GEMINI_API_KEY` | Gemini model calls | Required for AI itinerary generation in the current setup |
| `OPENAI_BASE_URL` | Google Gemini OpenAI-compatible endpoint | Use `https://generativelanguage.googleapis.com/v1beta/openai/` for the configured Gemini path |
| `DEFAULT_MODEL` | Model name used by the agents | Current configuration has used `gemini-3.8-flash`; verify that this model remains available to your API key |
| `SUPABASE_URL` | Supabase project URL | Required by current configuration validation |
| `SUPABASE_KEY` | Supabase project key | Required by current configuration validation; verify the app's actual database access separately |
| `TAVILY_API_KEY` | Destination web search | Optional feature key |
| `FIRECRAWL_API_KEY` | Scrape a selected web page | Optional feature key |
| `SERPAPI_API_KEY` | Google Flights/Hotels search snapshots | Optional; use only within the account's free quota and rules |
| `RAILRADAR_API_KEY` | Train timetable lookup | Optional; free-plan endpoint access and quota vary; not a promise of confirmed seats/fares |

Minimal `.env` shape (replace placeholders locally; never commit real values):

```dotenv
GEMINI_API_KEY=replace_with_your_key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
DEFAULT_MODEL=gemini-3.8-flash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=replace_with_your_supabase_key
TAVILY_API_KEY=replace_with_your_tavily_key
FIRECRAWL_API_KEY=replace_with_your_firecrawl_key
SERPAPI_API_KEY=replace_with_your_serpapi_key
RAILRADAR_API_KEY=replace_with_your_railradar_key
```

Leave optional variables out if you do not have that provider configured. Do not add an OpenAI paid key just to satisfy a legacy variable name; the current agent configuration should be using the Gemini key and Google base URL as described above.

## Free-tier and data-quality safeguards

- Do not add payment details to any provider when maintaining the project's ₹0/no-billing requirement.
- Read each provider's current plan, quota, and acceptable-use terms before enabling it. Free tiers can change.
- Avoid automatic retries for provider requests; inspect errors before deciding whether a single retry is safe.
- Cache identical searches where implemented, and don't trigger a search each time the page redraws.
- A successful API response is not the same as confirmed booking availability.
- Never fill missing flight, hotel, or transport results with hard-coded demo offers.
- Keep source links beside provider results so users can verify the result directly.

## Deploy to Render

1. Commit the intended source files and `render.yaml` to a GitHub repository you own.
2. Confirm `.env`, backups, repair scripts, and local credentials are not staged.
3. In Render, choose **New + → Blueprint** and connect the repository containing `render.yaml`.
4. Review the service plan and build/start commands before deploying.
5. Add secret environment values in Render's dashboard when prompted. The `render.yaml` should contain `sync: false` placeholders rather than secrets.
6. Open the generated service URL and test the comparison tab before making an AI itinerary request.
7. Review provider quotas and Render logs after any failure. Do not continuously retry a failed external search.

The included Render configuration uses a Streamlit start command and installs only runtime dependencies. Check the live `render.yaml` in your own repository before deploying; it must be present at the repository root and match the files that are actually committed.

## Security: never commit API keys

1. Store keys in `.env` for local development and in Render's dashboard for deployment.
2. Add `.env` and `.env.*` to `.gitignore`, while keeping `.env.example` trackable.
3. Commit only placeholders in `.env.example`—never real tokens, passwords, or Supabase credentials.
4. If a key was ever committed or shared, deleting the line is not enough. Revoke/rotate the key in the provider dashboard, then remove it from the repository history if necessary.
5. Before every commit, run:

```powershell
git check-ignore .env
git status --short
git diff --cached --check
git diff --cached --stat
```

`git check-ignore .env` should show `.env` as ignored. Inspect staged changes and confirm no secrets are present before committing.

## Temporary files to keep out of Git

These are local debugging artifacts, not application features:

- `.repair-backups/`
- `repair_*.py`
- `preflight_tavily_firecrawl.py`
- `integrate_tavily_firecrawl_once.py`
- `fix_web_research_test_isolation.py`
- `fix_redbus_direct_link.py`
- `repair_agent_logger_v1.py`
- `travel_planner/orchestration/core/graph_builder.py.bak`
- Local `.venv/`, Python cache files, logs, and `.env`

Keep the actual app and test files, including `web_app.py`, `travel_planner/utils/web_research_once.py`, `travel_planner/utils/live_travel_provider.py`, and the relevant tests. Do not delete backups until you've confirmed the project runs and the intended changes are committed; backups can be deleted afterward if you no longer need rollback copies.

##Deployed Link
https://ai-travel-planner-m0pk.onrender.com
## Known limitations

- No ticket, room, or bus-seat booking takes place inside this app.
- RailRadar integration is timetable-oriented; confirm fares and seats through official IRCTC before purchase.
- SerpApi returns search snapshots; a price can change or be unavailable by the time the user books.
- Bus search currently opens an external route-search page rather than returning a verified seat map.
- Supabase credential validation does not by itself prove database queries are working. Test persistence separately before relying on it.
- Render's free service has resource and inactivity limitations and may be slow on its first request after sleeping.

## Acknowledgements

This project started from [BjornMelin/openai-agents-travel-graph](https://github.com/BjornMelin/openai-agents-travel-graph) and has been adapted to use Gemini and a zero-cost, research-first comparison workflow. Review the upstream repository's license and preserve any required attribution/license files when redistributing your fork.
