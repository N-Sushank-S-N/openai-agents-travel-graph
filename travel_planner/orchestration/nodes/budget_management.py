"""
Budget management node implementation for the travel planning workflow.

This module defines functions for managing the travel budget using the budget
management agent, both for individual execution and as part of parallel processing.
"""

from travel_planner.agents.budget_management import BudgetManagementAgent
from travel_planner.data.models import BudgetSummary, TravelPlan
from travel_planner.data.models import AgentTaskParams
from travel_planner.orchestration.nodes.base_node import (
    execute_agent_task,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def budget_management(state: TravelPlanningState) -> TravelPlanningState:
    """Set a budget envelope from the user's request without using demo expenses."""
    query = state.query
    budget_range = getattr(query, "budget_range", None) if query is not None else None
    if not isinstance(budget_range, dict):
        budget_range = {}

    min_budget = budget_range.get("min")
    max_budget = budget_range.get("max", min_budget)
    try:
        min_budget = float(min_budget) if min_budget is not None else None
        max_budget = float(max_budget) if max_budget is not None else None
    except (TypeError, ValueError):
        min_budget, max_budget = None, None

    location_text = " ".join(
        str(getattr(query, name, "") or "")
        for name in ("origin", "destination", "raw_query")
    ).lower() if query is not None else ""
    india_places = (
        "india", "chennai", "vellore", "bengaluru", "bangalore", "mumbai",
        "delhi", "hyderabad", "kolkata", "pune", "kochi", "coimbatore",
        "madurai", "tirupati", "mysuru", "jaipur", "ahmedabad",
    )
    currency = "INR" if any(place in location_text for place in india_places) else "USD"

    if state.plan is None:
        state.plan = TravelPlan()

    if max_budget is not None and max_budget >= 0:
        state.plan.budget = BudgetSummary(
            total_budget=max_budget,
            currency=currency,
            spent=0.0,
            remaining=max_budget,
            breakdown={},
            items=[],
            notes=(
                f"Requested budget range: {min_budget if min_budget is not None else max_budget:g}–"
                f"{max_budget:g} {currency}. This is the budget ceiling from the request, "
                "not a verified quote; individual expenses need confirmation."
            ),
        )
        summary = f"Budget envelope set to {max_budget:g} {currency} (requested range {min_budget if min_budget is not None else max_budget:g}–{max_budget:g})."
    else:
        state.plan.budget = None
        summary = "No numeric budget range was supplied; no budget ceiling was assumed."

    state.update_stage(WorkflowStage.BUDGET_MANAGED)
    state.add_task_result(
        "budget_management",
        {"min_budget": min_budget, "max_budget": max_budget, "currency": currency, "source": "user query"},
    )
    state.conversation_history.append({"role": "system", "content": summary})
    logger.info(summary)
    return state


def budget_task(state: TravelPlanningState) -> dict[str, any]:
    """
    Execute budget management task (usually runs after other tasks are complete).

    Args:
        state: Current travel planning state

    Returns:
        Dictionary with task results
    """
    from travel_planner.orchestration.parallel import ParallelResult, ParallelTask

    try:
        agent = BudgetManagementAgent()
        result = agent.invoke(state)

        return {
            "result": ParallelResult(
                task_type=ParallelTask.BUDGET, result=result, completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in budget task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.BUDGET, result={}, error=str(e), completed=False
            )
        }
