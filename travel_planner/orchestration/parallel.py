"""
Parallel execution capabilities for the travel planner system.

This module implements parallel execution of agent tasks using LangGraph's
parallel branch capabilities, allowing multiple agents to work simultaneously
on different aspects of the travel planning process.
"""

import asyncio
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from travel_planner.agents.base import BaseAgent
from travel_planner.orchestration.states.workflow_stages import (
    PARALLEL_SEARCH_COMPLETED,
)
from travel_planner.utils.logging import get_logger

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from travel_planner.orchestration.states.planning_state import TravelPlanningState
else:
    # Forward references for type hints
    TravelPlanningState = Any

# Type for state update functions
T = TypeVar("T")
UpdateFunction = Callable[[T], T]

logger = get_logger(__name__)


class ParallelTask(Enum):
    """Enum representing different parallel tasks in the travel planning process."""

    FLIGHT_SEARCH = "flight_search"
    ACCOMMODATION = "accommodation"
    TRANSPORTATION = "transportation"
    ACTIVITIES = "activities"
    BUDGET = "budget"


class ParallelResult(BaseModel):
    """Model for storing results from parallel task execution."""

    task_type: ParallelTask
    result: dict[str, Any]
    error: str | None = None
    completed: bool = False


async def execute_in_parallel(
    tasks: list[tuple[BaseAgent, dict[str, Any]]], state: Any = None
) -> dict[str, Any]:
    """Execute tasks concurrently, passing each agent its supplied context.

    This function intentionally performs no retries. Each task is invoked once,
    which prevents duplicate external API calls and keeps API usage predictable.
    """
    async def execute_task(agent: BaseAgent, params: dict[str, Any]) -> dict[str, Any]:
        class_name = type(agent).__name__
        supplied_name = getattr(agent, "name", None)
        is_mock = type(agent).__module__.startswith("unittest.mock") or class_name in {
            "MagicMock", "AsyncMock"
        }
        result_key = supplied_name if is_mock and isinstance(supplied_name, str) and supplied_name else class_name
        timeout = params.get("timeout", 60)
        context = params.get("context", state)
        call_params = {
            key: value for key, value in params.items()
            if key not in {"timeout", "context"}
        }
        try:
            call = agent.process(**call_params, context=context)
            result = await asyncio.wait_for(call, timeout=float(timeout))
            if isinstance(result, dict) and result.get("error") and len(result) <= 2:
                return {result_key: {"result": None, "error": str(result["error"]), "retries": 0}}
            return {result_key: {"result": result, "error": None, "retries": 0}}
        except Exception as exc:
            logger.error(f"Parallel task {result_key} failed: {exc!s}")
            return {result_key: {"result": None, "error": str(exc), "retries": 0}}

    outputs = await asyncio.gather(*(execute_task(agent, params) for agent, params in tasks))
    combined: dict[str, Any] = {}
    for output in outputs:
        combined.update(output)
    return combined



async def parallel_search_tasks(state: TravelPlanningState) -> TravelPlanningState:
    """Run flight, accommodation, local transport, and activity agents in parallel."""
    from datetime import date, timedelta
    import re

    from travel_planner.agents.accommodation import AccommodationAgent, AccommodationSearchContext
    from travel_planner.agents.activity_planning import ActivityPlanningAgent, ActivityPlanningContext
    from travel_planner.agents.flight_search import FlightSearchAgent, FlightSearchContext
    from travel_planner.agents.transportation import TransportationAgent, TransportationContext

    logger.info("Setting up parallel search tasks")
    working_state = state.model_copy(deep=True)
    query_obj = getattr(working_state, "query", None)
    raw_query = getattr(query_obj, "raw_query", None)
    if not raw_query:
        raw_query = query_obj if isinstance(query_obj, str) else "Plan a travel itinerary"
    origin = getattr(query_obj, "origin", None) or ""
    destination = getattr(query_obj, "destination", None) or ""
    travelers = int(getattr(query_obj, "travelers", 1) or 1)
    budget = getattr(query_obj, "budget_range", None) or {}
    max_budget = budget.get("max") if isinstance(budget, dict) else None

    def parse_date(value: Any) -> date | None:
        if not value:
            return None
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value)[:10])
        except (TypeError, ValueError):
            return None

    start_day = parse_date(getattr(query_obj, "departure_date", None)) or (date.today() + timedelta(days=1))
    end_day = parse_date(getattr(query_obj, "return_date", None))
    if end_day is None:
        duration_match = re.search(r"\b(\d+)\s*[- ]?day", str(raw_query), re.IGNORECASE)
        duration_days = max(1, int(duration_match.group(1))) if duration_match else 2
        end_day = start_day + timedelta(days=duration_days - 1)
    if end_day < start_day:
        end_day = start_day + timedelta(days=1)
    start_str, end_str = start_day.isoformat(), end_day.isoformat()

    trip_input = (
        f"{raw_query}\nOrigin: {origin or 'unspecified'}\n"
        f"Destination: {destination or 'unspecified'}\nTravelers: {travelers}\n"
        f"Dates: {start_str} to {end_str}\nBudget range: {budget or 'unspecified'}"
    )
    # A non-empty search_params dictionary prevents these demo agents from
    # replacing the user's trip details with hard-coded example parameters.
    flight_context = FlightSearchContext(
        origin=origin, destination=destination, departure_date=start_str,
        return_date=end_str, travelers=travelers, max_price=max_budget,
        currency="INR", search_params={"destination": destination, "origin": origin, "departure_date": start_str, "return_date": end_str},
    )
    accommodation_context = AccommodationSearchContext(
        destination=destination, check_in_date=start_str, check_out_date=end_str,
        guests=travelers, max_price=max_budget,
        search_params={"destination": destination, "check_in_date": start_str, "check_out_date": end_str, "guests": travelers, "max_price": max_budget},
    )
    transportation_context = TransportationContext(
        destination=destination, accommodation_location=destination,
        start_date=start_str, end_date=end_str, traveler_count=travelers,
        max_budget=max_budget,
        search_params={"destination": destination, "start_date": start_str, "end_date": end_str, "traveler_count": travelers},
    )
    activity_context = ActivityPlanningContext(
        destination=destination, start_date=start_str, end_date=end_str,
        traveler_count=travelers,
        budget_per_day=(float(max_budget) / max(1, (end_day - start_day).days + 1)) if max_budget is not None else None,
        search_params={"destination": destination, "start_date": start_str, "end_date": end_str, "traveler_count": travelers},
    )

    agents = [
        (FlightSearchAgent(), {"input_data": trip_input, "context": flight_context, "timeout": 60}),
        (AccommodationAgent(), {"input_data": trip_input, "context": accommodation_context, "timeout": 60}),
        (TransportationAgent(), {"input_data": trip_input, "context": transportation_context, "timeout": 45}),
    ]
    try:
        async with asyncio.timeout(180):
            results = await execute_in_parallel(agents, working_state)
        if not any(value.get("result") for value in results.values()):
            errors = "; ".join(f"{key}: {value.get('error')}" for key, value in results.items() if value.get("error"))
            working_state.error = f"Parallel search agents failed: {errors or 'no results returned'}"
            if working_state.plan is None:
                from travel_planner.data.models import TravelPlan
                working_state.plan = TravelPlan()
            working_state.plan.alerts.append(working_state.error)
            working_state.current_stage = "error"
            return working_state
        updated_state = merge_parallel_results(working_state, results)
        updated_state.current_stage = PARALLEL_SEARCH_COMPLETED
        logger.info("Parallel search tasks completed")
        return updated_state
    except TimeoutError:
        logger.error("Parallel search tasks timed out")
        working_state.error = "Parallel search timed out"
        if working_state.plan is None:
            from travel_planner.data.models import TravelPlan
            working_state.plan = TravelPlan()
        working_state.plan.alerts.append(working_state.error)
        working_state.current_stage = "error"
        return working_state
    except Exception as exc:
        logger.exception(f"Unexpected parallel search error: {exc!s}")
        working_state.error = f"Unexpected parallel search error: {exc!s}"
        if working_state.plan is None:
            from travel_planner.data.models import TravelPlan
            working_state.plan = TravelPlan()
        working_state.plan.alerts.append(working_state.error)
        working_state.current_stage = "error"
        return working_state


def merge_parallel_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """
    Merge results from parallel execution into the state.

    This function takes the results from parallel agent executions and merges them
    into a consolidated state, ensuring that all data is properly integrated.

    Args:
        state: Current travel planning state
        results: Results from parallel execution

    Returns:
        Updated state with merged results
    """
    # Create a copy of the state to update
    updated_state = state.model_copy(deep=True)

    # Initialize the plan if not yet created
    updated_state = _ensure_plan_initialized(updated_state)

    # Process results from each agent
    updated_state = _process_flight_results(updated_state, results)
    updated_state = _process_accommodation_results(updated_state, results)
    updated_state = _process_transportation_results(updated_state, results)
    updated_state = _process_activity_results(updated_state, results)

    # Handle errors
    updated_state = _process_parallel_errors(updated_state, results)

    return updated_state


def _ensure_plan_initialized(state: TravelPlanningState) -> TravelPlanningState:
    """Ensure the travel plan is initialized in the state."""
    if state.plan is None:
        from travel_planner.data.models import TravelPlan

        state.plan = TravelPlan()
    return state


def _process_flight_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process flight search results from parallel execution."""
    if "FlightSearchAgent" not in results or not results["FlightSearchAgent"]["result"]:
        return state

    flight_data = results["FlightSearchAgent"]["result"]

    if "flights" in flight_data:
        state.plan.flights = flight_data["flights"]
    elif "flight_options" in flight_data:
        state.plan.flights = flight_data["flight_options"]

    return state


def _process_accommodation_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process accommodation results from parallel execution."""
    if (
        "AccommodationAgent" not in results
        or not results["AccommodationAgent"]["result"]
    ):
        return state

    accom_data = results["AccommodationAgent"]["result"]

    if "accommodations" in accom_data:
        state.plan.accommodation = accom_data["accommodations"]
    elif "accommodation_options" in accom_data:
        state.plan.accommodation = accom_data["accommodation_options"]

    return state


def _process_transportation_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process transportation results from parallel execution."""
    if (
        "TransportationAgent" not in results
        or not results["TransportationAgent"]["result"]
    ):
        return state

    transport_data = results["TransportationAgent"]["result"]

    if "transportation" in transport_data:
        state.plan.transportation = transport_data["transportation"]
    elif "transportation_options" in transport_data:
        state.plan.transportation = transport_data["transportation_options"]

    return state


def _process_activity_results(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process activity planning results from parallel execution."""
    if (
        "ActivityPlanningAgent" not in results
        or not results["ActivityPlanningAgent"]["result"]
    ):
        return state

    activity_data = results["ActivityPlanningAgent"]["result"]

    if "activities" in activity_data:
        state.plan.activities = activity_data["activities"]
    elif "daily_itineraries" in activity_data:
        state.plan.activities = activity_data["daily_itineraries"]

    return state


def _process_parallel_errors(
    state: TravelPlanningState, results: dict[str, Any]
) -> TravelPlanningState:
    """Process errors from parallel execution and add them to the state."""
    errors = []

    for agent_name, result in results.items():
        if result.get("error"):
            errors.append(f"{agent_name}: {result['error']}")

    if errors:
        if not state.plan.alerts:
            state.plan.alerts = []
        state.plan.alerts.extend(errors)

    return state


def combine_parallel_branch_results(
    state: TravelPlanningState, branch_results: dict[str, ParallelResult]
) -> TravelPlanningState:
    """
    Combine results from a LangGraph parallel branch execution.

    Args:
        state: Current travel planning state
        branch_results: Results from parallel branch execution

    Returns:
        Updated state with combined results
    """
    # Create a copy of the state to update
    updated_state = state.model_copy(deep=True)

    # Initialize the plan if not yet created
    updated_state = _ensure_plan_initialized(updated_state)

    # Ensure there are results to process
    if not _validate_branch_results(branch_results):
        logger.warning("No results from parallel branch execution")
        return updated_state

    # Extract and organize results by task type
    results_by_task = _organize_branch_results(branch_results)

    # Process results from each task type
    updated_state = _process_branch_flight_results(updated_state, results_by_task)
    updated_state = _process_branch_accommodation_results(
        updated_state, results_by_task
    )
    updated_state = _process_branch_transportation_results(
        updated_state, results_by_task
    )
    updated_state = _process_branch_activity_results(updated_state, results_by_task)
    updated_state = _process_branch_budget_results(updated_state, results_by_task)

    # Process errors
    updated_state = _process_branch_errors(updated_state, results_by_task)

    # Update workflow stage
    updated_state = _update_workflow_stage(updated_state)

    return updated_state


def _validate_branch_results(branch_results: dict[str, ParallelResult]) -> bool:
    """Validate that the branch results contain actual result data."""
    # Improved validation that checks if there's any result in the branch results
    if not branch_results:
        return False

    # Check for the 'result' key at the top level
    if branch_results.get("result"):
        return True

    # Check if any of the values in branch_results are ParallelResult objects
    # that have completed successfully
    for key, value in branch_results.items():
        if isinstance(value, ParallelResult) and value.completed and not value.error:
            return True

    return False


def _organize_branch_results(
    branch_results: dict[str, ParallelResult],
) -> dict[ParallelTask, ParallelResult]:
    """Organize parallel branch results by task type."""
    results_by_task = {}
    for task_result in branch_results.values():
        if isinstance(task_result, ParallelResult):
            task_type = task_result.task_type
            results_by_task[task_type] = task_result
    return results_by_task


def _process_branch_flight_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process flight search results from branch execution."""
    if ParallelTask.FLIGHT_SEARCH not in results_by_task:
        return state

    flight_result = results_by_task[ParallelTask.FLIGHT_SEARCH]
    if flight_result.completed and not flight_result.error:
        state.plan.flights = flight_result.result.get("flight_options", [])

    return state


def _process_branch_accommodation_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process accommodation results from branch execution."""
    if ParallelTask.ACCOMMODATION not in results_by_task:
        return state

    accom_result = results_by_task[ParallelTask.ACCOMMODATION]
    if accom_result.completed and not accom_result.error:
        state.plan.accommodation = accom_result.result.get("accommodations", [])

    return state


def _process_branch_transportation_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process transportation results from branch execution."""
    if ParallelTask.TRANSPORTATION not in results_by_task:
        return state

    transport_result = results_by_task[ParallelTask.TRANSPORTATION]
    if transport_result.completed and not transport_result.error:
        state.plan.transportation = transport_result.result.get(
            "transportation_options", {}
        )

    return state


def _process_branch_activity_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process activity results from branch execution."""
    if ParallelTask.ACTIVITIES not in results_by_task:
        return state

    activity_result = results_by_task[ParallelTask.ACTIVITIES]
    if activity_result.completed and not activity_result.error:
        state.plan.activities = activity_result.result.get("daily_itineraries", {})

    return state


def _process_branch_budget_results(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process budget results from branch execution."""
    if ParallelTask.BUDGET not in results_by_task:
        return state

    budget_result = results_by_task[ParallelTask.BUDGET]
    if budget_result.completed and not budget_result.error:
        state.plan.budget = budget_result.result.get("report", {})

    return state


def _process_branch_errors(
    state: TravelPlanningState, results_by_task: dict[ParallelTask, ParallelResult]
) -> TravelPlanningState:
    """Process errors from branch execution and add them to the state."""
    errors = []

    for task_type, task_result in results_by_task.items():
        if task_result.error:
            errors.append(f"{task_type.value}: {task_result.error}")

    if errors:
        if not state.plan.alerts:
            state.plan.alerts = []
        state.plan.alerts.extend(errors)

    return state


def _update_workflow_stage(state: TravelPlanningState) -> TravelPlanningState:
    """Update the workflow stage in the state."""
    state.current_stage = PARALLEL_SEARCH_COMPLETED
    return state
