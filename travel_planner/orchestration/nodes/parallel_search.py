"""
Parallel search node implementation for the travel planning workflow.

Runs independent travel searches concurrently and combines their results.
"""

from travel_planner.orchestration.parallel import parallel_search_tasks
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def create_parallel_search_branch():
    """
    Return the existing async function that executes parallel searches.

    Kept under its original name so the graph builder can use it unchanged.
    """
    return parallel_search_tasks


def combine_search_results(state: TravelPlanningState) -> TravelPlanningState:
    """Record a summary after parallel searches finish."""

    logger.info("Combining results from parallel search")

    state.update_stage(WorkflowStage.PARALLEL_SEARCH_COMPLETED)

    has_plan = state.plan is not None

    flights = state.plan.flights if has_plan else None
    accommodations = state.plan.accommodation if has_plan else None
    transportation = state.plan.transportation if has_plan else None

    flight_count = len(flights) if flights else 0
    accommodation_count = len(accommodations) if accommodations else 0
    transportation_count = len(transportation) if transportation else 0

    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                f"Completed parallel search: {flight_count} flights, "
                f"{accommodation_count} accommodations, "
                f"{transportation_count} transportation options"
            ),
        }
    )

    return state