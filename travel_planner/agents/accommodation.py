"""
Accommodation Agent for the travel planner system.

This module implements the specialized agent responsible for searching,
comparing, and recommending accommodation options for the travel itinerary.
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


class AccommodationType(str, Enum):
    """Types of accommodation."""

    HOTEL = "hotel"
    APARTMENT = "apartment"
    HOSTEL = "hostel"
    RESORT = "resort"
    VILLA = "villa"
    GUESTHOUSE = "guesthouse"


@dataclass
class AccommodationOption:
    """A single accommodation option."""

    id: str
    name: str
    type: AccommodationType
    location: str
    price_per_night: float
    currency: str
    rating: float | None = None
    amenities: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    description: str = ""
    address: str = ""
    booking_url: str = ""
    refundable: bool = False
    reviews_count: int = 0

    @property
    def formatted_price(self) -> str:
        """Get the formatted price with currency symbol."""
        if self.currency == "USD":
            return f"${self.price_per_night:.2f}"
        elif self.currency == "EUR":
            return f"€{self.price_per_night:.2f}"
        else:
            return f"{self.price_per_night:.2f} {self.currency}"


class AccommodationSearchContext(AgentContext):
    """Context for the accommodation search agent."""

    destination: str = ""
    check_in_date: str | None = None
    check_out_date: str | None = None
    guests: int = 1
    rooms: int = 1
    accommodation_type: AccommodationType | None = None
    max_price: float | None = None
    amenities: list[str] = Field(default_factory=list)
    accommodation_options: list[AccommodationOption] = Field(default_factory=list)
    selected_accommodation: AccommodationOption | None = None
    search_params: dict[str, Any] = Field(default_factory=dict)
    search_results_raw: dict[str, Any] = Field(default_factory=dict)


class AccommodationAgent(BaseAgent[AccommodationSearchContext]):
    """
    Specialized agent for accommodation search and booking.

    This agent is responsible for:
    1. Searching multiple accommodation booking sites for optimal options
    2. Filtering results based on user preferences
    3. Providing comparisons and recommendations
    4. Supporting booking capabilities when needed
    5. Presenting options with key details
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the accommodation search agent.

        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="accommodation_agent",
            instructions="""
            You are a specialized agent focused on finding the best accommodation 
            options.
            Your goal is to research, compare, and recommend accommodations that match
            the traveler's preferences and budget. You should consider factors like
            location, amenities, reviews, and value.
            """,
            model="gpt-4o",
            tools=[],  # No tools initially, they would be added in a real implementation
        )

        config = config or default_config
        super().__init__(config, AccommodationSearchContext)

        # Add tools for specific accommodation search functionality
        # These would typically be implemented as part of the full system
        # self.add_tool(search_accommodations)
        # self.add_tool(compare_prices)
        # self.add_tool(check_availability)
        # self.add_tool(filter_by_amenities)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: AccommodationSearchContext | None = None,
    ) -> Any:
        """
        Run the accommodation search agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional accommodation search context

        Returns:
            Updated context and accommodation search results
        """
        try:
            # Initialize context if not provided
            if not context:
                context = AccommodationSearchContext()

            # Process the input
            result = await self.process(input_data, context)
            return result
        except Exception as e:
            error_msg = f"Error in accommodation search agent: {e!s}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def process(self, input_data, context):
        """Search without fabricating hotel offers or making redundant model calls."""
        if context is None:
            context = AccommodationSearchContext()
        if not context.search_params:
            await self._extract_search_parameters(input_data, context)
        search_results = await self._search_accommodations(context)
        ranked_options = await self._rank_accommodation_options(search_results, context)
        context.accommodation_options = ranked_options
        if ranked_options:
            summary = await self._generate_options_summary(ranked_options, context)
        else:
            summary = (
                f"No verified accommodation offers are available for {context.destination or 'this destination'}. "
                "Configure a real accommodation-search provider."
            )
        return {"context": context, "accommodations": ranked_options, "summary": summary}

    async def _extract_search_parameters(self, input_data, context):
        """Use only request/context fields; do not overwrite them with demo defaults."""
        text = input_data if isinstance(input_data, str) else self._get_latest_user_input(input_data)
        for line in text.splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key, value = key.strip().lower(), value.strip()
            if key == "destination" and value and not context.destination:
                context.destination = value
            elif key in {"check-in", "check in", "check-in date"} and value and not context.check_in_date:
                context.check_in_date = value
            elif key in {"check-out", "check out", "check-out date"} and value and not context.check_out_date:
                context.check_out_date = value
            elif key == "travelers" and value:
                try:
                    context.guests = int(value)
                except ValueError:
                    pass
        context.search_params = {
            "destination": context.destination,
            "check_in_date": context.check_in_date,
            "check_out_date": context.check_out_date,
            "guests": context.guests,
            "rooms": context.rooms,
            "accommodation_type": context.accommodation_type.value if context.accommodation_type else None,
            "max_price": context.max_price,
            "amenities": list(context.amenities),
        }

    async def _search_accommodations(self, context):
        """Return no offers until a real accommodation-search integration is configured."""
        logger.warning(
            "No live accommodation provider configured; refusing to return demo hotel offers."
        )
        context.search_results_raw = {"results": []}
        return []

    async def _rank_accommodation_options(
        self, search_results: list[dict[str, Any]], context: AccommodationSearchContext
    ) -> list[AccommodationOption]:
        """
        Rank and convert accommodation options based on user preferences.

        Args:
            search_results: Raw accommodation search results
            context: Accommodation search context

        Returns:
            List of ranked AccommodationOption objects
        """
        accommodation_options = []

        # Convert raw search data to AccommodationOption objects
        for result in search_results:
            option = AccommodationOption(
                id=result["id"],
                name=result["name"],
                type=AccommodationType(result["type"]),
                location=result["location"],
                price_per_night=result["price_per_night"],
                currency=result["currency"],
                rating=result.get("rating"),
                amenities=result.get("amenities", []),
                images=result.get("images", []),
                description=result.get("description", ""),
                address=result.get("address", ""),
                booking_url=result.get("booking_url", ""),
                refundable=result.get("refundable", False),
                reviews_count=result.get("reviews_count", 0),
            )
            accommodation_options.append(option)

        # Apply any ranking or filtering based on user preferences
        # For this demo, we'll sort by rating (highest first), then by price
        # (lowest first)
        accommodation_options.sort(
            key=lambda x: (-x.rating if x.rating else 0, x.price_per_night)
        )

        return accommodation_options[:5]  # Return top 5 options

    async def _generate_options_summary(
        self, options: list[AccommodationOption], context: AccommodationSearchContext
    ) -> str:
        """
        Generate a human-readable summary of accommodation options.

        Args:
            options: List of accommodation options
            context: Accommodation search context

        Returns:
            Summary text
        """
        # Prepare a specific prompt for generating a summary
        summary_prompt = (
            "Create a summary of the following accommodation options for "
            "{destination}. "
            "Highlight key features, price differences, and which options best match "
            "the traveler's preferences for {amenities}. Max budget is "
            "{max_price} {currency} per night. "
            "Stay dates: {check_in} to {check_out}. "
            "\n\nOptions:\n\n{options_text}"
        )

        # Prepare options in a format the model can understand
        options_text = "\n\n".join(
            [
                f"Option {i + 1}: {option.name}\n"
                f"Type: {option.type.value}\n"
                f"Location: {option.location}\n"
                f"Price: {option.formatted_price} per night\n"
                f"Rating: {option.rating} ({option.reviews_count} reviews)\n"
                f"Amenities: {', '.join(option.amenities)}\n"
                f"Description: {option.description}\n"
                f"Refundable: {'Yes' if option.refundable else 'No'}"
                for i, option in enumerate(options[:5])  # Limit to top 5 options
            ]
        )

        messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": summary_prompt.format(
                    destination=context.destination,
                    amenities=", ".join(context.amenities),
                    max_price=context.max_price,
                    currency=options[0].currency if options else "EUR",
                    check_in=context.check_in_date,
                    check_out=context.check_out_date,
                    options_text=options_text,
                ),
            },
        ]

        response = await self._call_model(messages)

        # Return the generated summary
        return response.get("content", "")

    def _format_accommodation_option(
        self, option: AccommodationOption
    ) -> dict[str, Any]:
        """
        Format an accommodation option for display.

        Args:
            option: AccommodationOption object

        Returns:
            Formatted accommodation option dictionary
        """
        return {
            "id": option.id,
            "name": option.name,
            "type": option.type.value,
            "location": option.location,
            "price": option.formatted_price,
            "rating": f"{option.rating}/5" if option.rating else "Not rated",
            "amenities": option.amenities,
            "description": option.description,
            "refundable": option.refundable,
        }

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
