"""Wrapper that persists the finished plan after the original final-plan node."""
from __future__ import annotations
import inspect
from typing import Any
from travel_planner.orchestration.nodes.final_plan import generate_final_plan as _generate_final_plan
from travel_planner.data.travel_persistence import persist_travel_state


def _state_mapping(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        return dict(state)
    if hasattr(state, "model_dump"):
        try:
            return state.model_dump(mode="python")
        except TypeError:
            return state.model_dump()
    if hasattr(state, "__dict__"):
        return {key: value for key, value in vars(state).items() if not key.startswith("_")}
    return {}


async def generate_final_plan(state: Any) -> Any:
    """Persist the merged completed state without altering LangGraph's node result."""
    result = _generate_final_plan(state)
    if inspect.isawaitable(result):
        result = await result

    # LangGraph nodes often return partial update dicts. Merge them for persistence,
    # but return exactly the original node result to preserve the graph contract.
    if isinstance(result, dict):
        persist_state = _state_mapping(state)
        persist_state.update(result)
    elif result is None:
        persist_state = state
    else:
        persist_state = result

    await persist_travel_state(persist_state)
    return state if result is None else result
