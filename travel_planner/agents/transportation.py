"""
Transportation Agent for the travel planner system.

This module implements the specialized agent responsible for planning
local transportation, researching rental options, and optimizing
travel routes for the travel itinerary.
"""

import re
from dataclasses import dataclass, field
from pydantic import Field
from enum import Enum
from typing import Any

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils.error_handling import with_retry
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class TransportationType(str, Enum):
    """Types of transportation."""

    RENTAL_CAR = "rental_car"
    PUBLIC_TRANSIT = "public_transit"
    TAXI = "taxi"
    RIDESHARE = "rideshare"
    SHUTTLE = "shuttle"
    FERRY = "ferry"
    TRAIN = "train"
    BUS = "bus"
    WALKING = "walking"
    BICYCLE = "bicycle"


@dataclass
class TransportationOption:
    """A single transportation option."""

    id: str
    type: TransportationType
    name: str
    price: float
    currency: str
    duration_minutes: int
    start_location: str
    end_location: str
    provider: str = ""
    details: str = ""
    booking_url: str = ""
    schedule: dict[str, Any] | None = None
    amenities: list[str] = field(default_factory=list)
    accessibility: list[str] = field(default_factory=list)
    eco_friendly: bool = False

    @property
    def formatted_price(self) -> str:
        """Get the formatted price with currency symbol."""
        if self.currency == "USD":
            return f"${self.price:.2f}"
        elif self.currency == "EUR":
            return f"€{self.price:.2f}"
        else:
            return f"{self.price:.2f} {self.currency}"

    @property
    def formatted_duration(self) -> str:
        """Get the formatted duration as hours and minutes."""
        hours, minutes = divmod(self.duration_minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


class TransportationContext(AgentContext):
    """Context for the transportation agent."""

    destination: str = ""
    accommodation_location: str = ""
    start_date: str | None = None
    end_date: str | None = None
    traveler_count: int = 1
    preferred_transportation_types: list[TransportationType] = field(
        default_factory=list
    )
    has_children: bool = False
    has_accessibility_needs: bool = False
    max_budget: float | None = None
    transportation_options: dict[str, list[TransportationOption]] = field(
        default_factory=dict
    )
    selected_options: dict[str, TransportationOption] = Field(default_factory=dict)
    search_params: dict[str, Any] = Field(default_factory=dict)
    points_of_interest: list[str] = Field(default_factory=list)


class TransportationAgent(BaseAgent[TransportationContext]):
    """
    Specialized agent for transportation planning.

    This agent is responsible for:
    1. Planning airport transfers and local transportation
    2. Researching rental car options
    3. Identifying public transit and transportation options
    4. Optimizing transportation for convenience and cost
    5. Considering special needs and requirements
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the transportation agent.

        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="transportation_agent",
            instructions="""
            You are a specialized transportation planner agent.
            Your goal is to research, compare, and recommend the best transportation
            options for travelers at their destination. Consider factors like
            convenience, cost, travel time, and special requirements.
            """,
            model="gpt-4o",
            tools=[],  # No tools initially, they would be added in a real implementation
        )

        config = config or default_config
        super().__init__(config, TransportationContext)

        # Add tools for specific transportation search functionality
        # These would typically be implemented as part of the full system
        # self.add_tool(search_rental_cars)
        # self.add_tool(find_public_transit)
        # self.add_tool(calculate_transfer_options)
        # self.add_tool(optimize_route)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: TransportationContext | None = None,
    ) -> Any:
        """
        Run the transportation agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional transportation context

        Returns:
            Updated context and transportation planning results
        """
        try:
            # Initialize context if not provided
            if not context:
                context = TransportationContext()

            # Process the input
            result = await self.process(input_data, context)
            return result
        except Exception as e:
            error_msg = f"Error in transportation agent: {e!s}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def process(self, input_data, context):
        """Return only verified transportation data; never invent Paris-based prices."""
        if not context.search_params:
            await self._extract_transportation_requirements(input_data, context)
        context.transportation_options = {"airport_transfers": [], "local": [], "points_of_interest": {}}
        context.selected_options = context.selected_options or {}
        summary = (
            f"No verified transportation prices or routes are available for {context.destination or 'this destination'}. "
            "Use destination-specific transit or routing data before booking."
        )
        return {
            "context": context,
            "transportation_options": context.transportation_options,
            "selected_options": context.selected_options,
            "plan_summary": summary,
        }

    async def _extract_transportation_requirements(self, input_data, context):
        """Preserve explicit request fields without inventing a city, dates, or prices."""
        text = input_data if isinstance(input_data, str) else self._get_latest_user_input(input_data)
        for line in text.splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key, value = key.strip().lower(), value.strip()
            if key == "destination" and value and not context.destination:
                context.destination = value
            elif key == "travelers" and value:
                try:
                    context.traveler_count = int(value)
                except ValueError:
                    pass
            elif key == "dates" and value:
                dates = re.findall(r"\d{4}-\d{2}-\d{2}", value)
                if dates and not context.start_date:
                    context.start_date = dates[0]
                if len(dates) > 1 and not context.end_date:
                    context.end_date = dates[1]
        context.search_params = {
            "destination": context.destination,
            "start_date": context.start_date,
            "end_date": context.end_date,
            "traveler_count": context.traveler_count,
            "preferred_transportation_types": [t.value for t in context.preferred_transportation_types],
            "points_of_interest": list(context.points_of_interest),
            "max_budget": context.max_budget,
        }

    async def _plan_airport_transfers(self, context):
        """No airport transfers are returned without a destination-specific provider."""
        logger.warning("No verified airport-transfer provider configured; returning no offers.")
        return []

    async def _research_local_transportation(self, context):
        """No local transport offers are returned without a destination-specific provider."""
        logger.warning("No verified local-transport provider configured; returning no offers.")
        return []

    async def _plan_poi_transportation(self, context):
        """Return no priced route options until routing data is integrated."""
        return {}

    async def _generate_transportation_plan(
        self, context: TransportationContext
    ) -> str:
        """
        Generate a comprehensive transportation plan.

        Args:
            context: Transportation context

        Returns:
            Transportation plan text
        """
        # Prepare a specific prompt for generating a plan
        plan_prompt = (
            "Create a detailed transportation plan for a trip to {destination} "
            "staying at {accommodation}. The trip is from {start_date} to {end_date} "
            "for {traveler_count} traveler(s).\n\n"
            "Please include:\n"
            "1. Airport transfer recommendations\n"
            "2. Best local transportation options\n"
            "3. Specific routes to these points of interest: {pois}\n"
            "4. Cost breakdown and budget analysis\n"
            "5. Tips for navigating local transportation\n\n"
            "Transportation preferences: {preferences}\n"
            "Max budget: {budget} EUR\n"
            "{accessibility}"
        )

        accessibility_note = ""
        if context.has_children:
            accessibility_note += "Note: Traveling with children. "
        if context.has_accessibility_needs:
            accessibility_note += "Note: Has accessibility requirements. "

        messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": plan_prompt.format(
                    destination=context.destination,
                    accommodation=context.accommodation_location,
                    start_date=context.start_date,
                    end_date=context.end_date,
                    traveler_count=context.traveler_count,
                    pois=", ".join(context.points_of_interest),
                    preferences=", ".join(
                        [t.value for t in context.preferred_transportation_types]
                    ),
                    budget=context.max_budget,
                    accessibility=accessibility_note,
                ),
            },
        ]

        response = await self._call_model(messages)

        # Return the generated plan
        return response.get("content", "")

    def _get_latest_user_input(self, messages: list[dict[str, Any]]) -> str:
        """
        Extract the latest user input from a list of messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Latest user input text
        """
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    @with_retry(max_attempts=3)
    async def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call the OpenAI API with the given messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Model response
        """
        # Log inputs for debugging
        logger.debug(f"Calling model with messages: {messages}")

        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Log the response
        logger.debug(f"Model response: {response}")

        # Extract the content from the response
        content = response.choices[0].message.content
        return {"content": content}
