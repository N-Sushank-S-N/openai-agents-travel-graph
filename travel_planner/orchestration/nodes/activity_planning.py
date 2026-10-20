"""
Activity planning node implementation for the travel planning workflow.

This module defines functions for planning activities using the activity
planning agent, both for individual execution and as part of parallel processing.
"""

from travel_planner.agents.activity_planning import ActivityPlanningAgent
from travel_planner.data.models import AgentTaskParams
from travel_planner.orchestration.nodes.base_node import (
    execute_agent_task,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def activity_planning(state: TravelPlanningState) -> TravelPlanningState:
    """Mark the activity stage complete without duplicating the orchestrator itinerary call.

    Structured activity listings are intentionally left empty until a real activity
    provider is configured. The orchestrator's generated itinerary narrative is kept.
    """
    if state is not None:
        state.update_stage(WorkflowStage.ACTIVITIES_PLANNED)
    return state

def activities_task(state: TravelPlanningState) -> dict[str, any]:
    """
    Execute activity planning task in parallel branch.

    Args:
        state: Current travel planning state

    Returns:
        Dictionary with task results
    """
    from travel_planner.orchestration.parallel import ParallelResult, ParallelTask

    try:
        agent = ActivityPlanningAgent()
        result = agent.invoke(state)

        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACTIVITIES, result=result, completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in activities task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACTIVITIES,
                result={},
                error=str(e),
                completed=False,
            )
        }
