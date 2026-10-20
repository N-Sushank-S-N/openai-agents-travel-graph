"""Finalization node for an already generated travel-planning response."""
from travel_planner.data.models import TravelPlan
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def generate_final_plan(state: TravelPlanningState) -> TravelPlanningState:
    """Finalize the current plan without another model request."""
    logger.info("Finalizing travel plan from existing workflow results")
    if state.plan is None:
        state.plan = TravelPlan()

    # The query-analysis agent already generated a complete narrative response.
    answer = ""
    for message in reversed(state.conversation_history or []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                answer = content.strip()
                break

    if answer and not state.plan.overview:
        state.plan.overview = answer

    query = state.query
    destination = getattr(query, "destination", None) if query is not None else None
    if destination and not state.plan.destination:
        state.plan.destination = {"name": destination, "source": "user-provided query"}

    if not answer and not state.plan.overview:
        state.plan.overview = (
            "A complete narrative could not be generated in this run. "
            "Structured results may be incomplete; verify all details before travel."
        )
        state.plan.alerts.append("No assistant itinerary text was available during finalization.")

    state.update_stage(WorkflowStage.COMPLETE)
    result = {"reused_existing_answer": bool(answer), "destination": destination}
    state.add_task_result("generate_final_plan", result)
    state.conversation_history.append(
        {"role": "system", "content": "Finalization completed using the existing itinerary response; no extra model call was made."}
    )

    # Best-effort checkpoint only; the itinerary should remain usable if checkpoint
    # serialization hits the repository's existing data-model inconsistencies.
    try:
        from travel_planner.orchestration.serialization.checkpoint import save_state_checkpoint
        checkpoint_id = save_state_checkpoint(state)
        if hasattr(state, "saved_checkpoint_id"):
            state.saved_checkpoint_id = checkpoint_id
        elif hasattr(state, "checkpoint_id"):
            state.checkpoint_id = checkpoint_id
    except Exception as exc:
        logger.warning(f"Checkpoint skipped because state serialization failed: {exc!s}")

    logger.info("Finalization completed")
    return state
