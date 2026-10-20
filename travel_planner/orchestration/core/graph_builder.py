"""
Graph builder for the travel planner workflow.

Connects the main travel-planning nodes using LangGraph.
"""

from langgraph.graph import END, START, StateGraph

from travel_planner.orchestration.nodes.activity_planning import activity_planning
from travel_planner.orchestration.nodes.budget_management import budget_management
from travel_planner.orchestration.nodes.destination_research import (
    destination_research,
)
from travel_planner.orchestration.nodes.final_plan import generate_final_plan
from travel_planner.orchestration.nodes.parallel_search import (
    combine_search_results,
    create_parallel_search_branch,
)
from travel_planner.orchestration.nodes.query_analysis import query_analysis
from travel_planner.orchestration.routing.conditions import query_research_needed
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def create_planning_graph() -> StateGraph:
    """Build and compile the main travel-planning workflow."""

    logger.info("Creating planning graph")

    workflow = StateGraph(TravelPlanningState)

    # Register the workflow nodes.
    workflow.add_node("analyze_query", query_analysis)
    workflow.add_node("research_destination", destination_research)
    workflow.add_node("parallel_search", create_parallel_search_branch())
    workflow.add_node("combine_search_results", combine_search_results)
    workflow.add_node("plan_activities", activity_planning)
    workflow.add_node("manage_budget", budget_management)
    workflow.add_node("generate_final_plan", generate_final_plan)

    # Begin by analysing the user's query.
    workflow.add_edge(START, "analyze_query")

    # Research the destination if the query doesn't specify one.
    workflow.add_conditional_edges(
        "analyze_query",
        query_research_needed,
        {
            "research_destination": "research_destination",
            "flight_search": "parallel_search",
        },
    )

    # Continue through the main planning stages.
    workflow.add_edge("research_destination", "parallel_search")
    workflow.add_edge("parallel_search", "combine_search_results")
    workflow.add_edge("combine_search_results", "plan_activities")
    workflow.add_edge("plan_activities", "manage_budget")
    workflow.add_edge("manage_budget", "generate_final_plan")
    workflow.add_edge("generate_final_plan", END)

    logger.info("Planning graph created and compiled")

    return workflow.compile()