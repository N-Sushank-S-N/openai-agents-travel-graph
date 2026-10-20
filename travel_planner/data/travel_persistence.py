"""Best-effort Supabase persistence for completed LangGraph travel plans."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _jsonable({k: v for k, v in vars(value).items() if not k.startswith("_")})
    return str(value)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _persist_sync(state: Any) -> bool:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        logger.info("Supabase persistence skipped: SUPABASE_URL/key not configured")
        return False
    try:
        from supabase import create_client
    except ImportError:
        logger.warning("Supabase persistence skipped: supabase package is not installed")
        return False

    query = _get(state, "query")
    # The project state model calls this field `travel_plan`; accept `plan` as a compatibility alias.
    plan = _get(state, "travel_plan")
    if plan is None:
        plan = _get(state, "plan")
    if plan is None:
        logger.info("Supabase persistence skipped: final state has no travel_plan/plan field")
        return False
    raw_key = _get(state, "saved_checkpoint_id") or _get(state, "session_id") or _get(state, "start_time")
    session_key = str(raw_key or uuid.uuid4())
    query_json = _jsonable(query)
    plan_json = _jsonable(plan)
    destination_value = _get(query, "destination") or _get(plan, "destination")
    if isinstance(destination_value, dict):
        destination_value = destination_value.get("name") or json.dumps(destination_value, ensure_ascii=False)
    origin_value = _get(query, "origin")
    preferences_value = (
        _get(query, "preferences")
        or _get(state, "preferences")
        or _get(state, "user_preferences")
    )
    payload = {
        "session_key": session_key,
        "origin": str(origin_value) if origin_value else None,
        "destination": str(destination_value) if destination_value else None,
        "request": query_json,
        "preferences": _jsonable(preferences_value),
        "plan": plan_json,
    }
    client = create_client(url, key)
    # A dedicated table avoids assuming the original app's normalized schema.
    client.table(os.getenv("SUPABASE_TRAVEL_PLANS_TABLE", "travel_plan_runs")).upsert(
        payload, on_conflict="session_key"
    ).execute()
    return True


async def persist_travel_state(state: Any) -> Any:
    """Save a completed plan when Supabase is configured; never fail planning on a save error."""
    if not os.getenv("SUPABASE_URL") or not (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    ):
        return state
    try:
        saved = await asyncio.to_thread(_persist_sync, state)
        if saved:
            logger.info("Travel plan saved to Supabase travel_plan_runs table")
    except Exception as exc:
        logger.exception("Unable to persist travel plan to Supabase (%s); continuing without persistence", type(exc).__name__)
    return state
