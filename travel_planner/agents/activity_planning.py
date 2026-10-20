"""
Activity Planning Agent for the travel planner system.

This module implements the specialized agent responsible for researching,
scheduling, and recommending activities and attractions for the travel itinerary.
"""

import re
from dataclasses import dataclass, field
from pydantic import Field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils.error_handling import with_retry
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)

# Constants
MAX_ACTIVITIES_PER_DAY = 3


class ActivityType(str, Enum):
    """Types of activities."""

    ATTRACTION = "attraction"
    TOUR = "tour"
    MUSEUM = "museum"
    OUTDOOR = "outdoor"
    FOOD = "food_and_drink"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    CULTURAL = "cultural"
    ADVENTURE = "adventure"
    WELLNESS = "wellness"


class WeatherCondition(str, Enum):
    """Types of weather conditions."""

    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    WINDY = "windy"
    HOT = "hot"
    COLD = "cold"


@dataclass
class Activity:
    """A single activity or attraction."""

    id: str
    name: str
    type: ActivityType
    location: str
    description: str
    price: float
    currency: str
    duration_minutes: int
    opening_hours: dict[str, dict[str, time]] = field(default_factory=dict)
    booking_required: bool = False
    booking_url: str = ""
    weather_dependent: bool = False
    suitable_weather: list[WeatherCondition] = field(default_factory=list)
    rating: float | None = None
    reviews_count: int = 0
    images: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    accessibility_features: list[str] = field(default_factory=list)

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


@dataclass
class ScheduledActivity:
    """An activity scheduled for a specific date and time."""

    activity: Activity
    date: str
    start_time: str
    end_time: str
    notes: str = ""


@dataclass
class DailyItinerary:
    """Itinerary for a single day."""

    date: str
    activities: list[ScheduledActivity] = field(default_factory=list)
    weather_forecast: dict[str, Any] | None = None
    notes: str = ""
    total_cost: float = 0.0
    currency: str = "EUR"


class ActivityPlanningContext(AgentContext):
    """Context for the activity planning agent."""

    destination: str = ""
    start_date: str | None = None
    end_date: str | None = None
    traveler_count: int = 1
    interests: list[str] = Field(default_factory=list)
    has_children: bool = False
    has_accessibility_needs: bool = False
    budget_per_day: float | None = None
    accommodation_location: str = ""
    available_activities: list[Activity] = Field(default_factory=list)
    daily_itineraries: dict[str, DailyItinerary] = Field(default_factory=dict)
    weather_forecasts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    search_params: dict[str, Any] = Field(default_factory=dict)
    excluded_activity_types: list[ActivityType] = Field(default_factory=list)


class ActivityPlanningAgent(BaseAgent[ActivityPlanningContext]):
    """
    Specialized agent for activity planning.

    This agent is responsible for:
    1. Researching activities and attractions at the destination
    2. Creating daily itineraries with scheduled activities
    3. Considering factors like opening hours, travel time, and weather
    4. Balancing must-see attractions with personalized experiences
    5. Ensuring activities are within budget constraints
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the activity planning agent.

        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="activity_planning_agent",
            instructions="""
            You are a specialized activity planning agent.
            Your goal is to research, recommend, and schedule activities and attractions
            that match the traveler's interests and preferences. Create logical daily
            itineraries that account for location, opening hours, travel time, 
            and budget.
            """,
            model="gpt-4o",
            tools=[],  # No tools initially, they would be added in a real implementation
        )

        config = config or default_config
        super().__init__(config, ActivityPlanningContext)

        # Add tools for specific activity planning functionality
        # These would typically be implemented as part of the full system
        # self.add_tool(search_activities)
        # self.add_tool(check_opening_hours)
        # self.add_tool(get_weather_forecast)
        # self.add_tool(calculate_travel_time)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: ActivityPlanningContext | None = None,
    ) -> Any:
        """
        Run the activity planning agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional activity planning context

        Returns:
            Updated context and activity planning results
        """
        try:
            # Initialize context if not provided
            if not context:
                context = ActivityPlanningContext()

            # Process the input
            result = await self.process(input_data, context)
            return result
        except Exception as e:
            error_msg = f"Error in activity planning agent: {e!s}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def process(self, input_data, context):
        """Return an honest empty structured result until live activity research is integrated."""
        if context is None:
            context = ActivityPlanningContext()
        if not context.search_params:
            await self._extract_activity_preferences(input_data, context)
        context.available_activities = []
        context.daily_itineraries = {}
        return {
            "context": context,
            "daily_itineraries": {},
            "summary": (
                f"No verified structured activity listings are available for {context.destination or 'this destination'}. "
                "The main itinerary narrative may still contain suggested places; verify opening times and prices separately."
            ),
        }

    async def _extract_activity_preferences(self, input_data, context):
        """Preserve supplied destination/trip details; no hard-coded demo destination."""
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
            "interests": list(context.interests),
            "budget_per_day": context.budget_per_day,
        }

    async def _research_activities(self, context):
        """Return no fabricated attraction entries when no live research provider is configured."""
        logger.warning("No live activity-search provider configured; returning no demo attractions.")
        return []

    async def _get_weather_forecasts(
        self, context: ActivityPlanningContext
    ) -> dict[str, dict[str, Any]]:
        """
        Get weather forecasts for the destination during the trip dates.

        Args:
            context: Activity planning context

        Returns:
            Dictionary mapping dates to weather forecasts
        """
        # In a real implementation, this would call a weather API
        # For demonstration, we'll create some mock forecasts

        forecasts = {}

        if not context.start_date or not context.end_date:
            return forecasts

        start = datetime.strptime(context.start_date, "%Y-%m-%d")
        end = datetime.strptime(context.end_date, "%Y-%m-%d")
        current_date = start

        # Generate forecasts for each day
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")

            # Create a mock forecast (in a real implementation, this would come from an API)
            # We'll alternate between sunny and cloudy for simplicity
            is_sunny = (current_date - start).days % 2 == 0

            forecasts[date_str] = {
                "condition": WeatherCondition.SUNNY
                if is_sunny
                else WeatherCondition.CLOUDY,
                "temperature_celsius": 22 if is_sunny else 19,
                "temperature_fahrenheit": 72 if is_sunny else 66,
                "precipitation_chance": 10 if is_sunny else 40,
                "wind_speed_kmh": 10 if is_sunny else 15,
            }

            current_date += timedelta(days=1)

        return forecasts

    async def _create_daily_itineraries(
        self, context: ActivityPlanningContext
    ) -> dict[str, DailyItinerary]:
        """
        Create daily itineraries for the trip dates.

        Args:
            context: Activity planning context

        Returns:
            Dictionary mapping dates to daily itineraries
        """
        if (
            not context.start_date
            or not context.end_date
            or not context.available_activities
        ):
            return {}

        itineraries = {}

        start = datetime.strptime(context.start_date, "%Y-%m-%d")
        end = datetime.strptime(context.end_date, "%Y-%m-%d")
        current_date = start

        # Create an itinerary for each day
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            day_of_week = current_date.strftime("%A")

            # Get weather forecast if available
            weather = context.weather_forecasts.get(date_str)

            # Create a new daily itinerary
            itinerary = DailyItinerary(
                date=date_str, weather_forecast=weather, currency="EUR"
            )

            # Filter activities based on day of week (opening hours) and weather
            suitable_activities = []
            for activity in context.available_activities:
                # Check if the activity is open on this day
                opening_hours = activity.opening_hours.get(day_of_week, {})
                if opening_hours and (
                    opening_hours.get("open") != time(0, 0)
                    or opening_hours.get("close") != time(0, 0)
                ):
                    # Check if the activity is suitable for the weather
                    if not activity.weather_dependent or not weather:
                        suitable_activities.append(activity)
                    elif (
                        weather
                        and weather.get("condition") in activity.suitable_weather
                    ):
                        suitable_activities.append(activity)

            # For a real implementation, we would create a logical daily schedule based on:
            # - Location proximity (to minimize travel time)
            # - Opening hours
            # - Activity durations
            # - Budget constraints

            # For this demo, we'll create a simple schedule with morning, afternoon, and evening activities

            # Morning activity (9:00 - 12:00)
            morning_activities = [
                a
                for a in suitable_activities
                if a.type
                in [ActivityType.MUSEUM, ActivityType.ATTRACTION, ActivityType.TOUR]
            ]
            if (
                morning_activities
                and len(itinerary.activities) < MAX_ACTIVITIES_PER_DAY
            ):
                selected = morning_activities[
                    0
                ]  # In a real implementation, we would make a smarter selection
                suitable_activities.remove(selected)

                itinerary.activities.append(
                    ScheduledActivity(
                        activity=selected,
                        date=date_str,
                        start_time="09:00",
                        end_time="12:00",
                        notes="Visit in the morning to avoid crowds",
                    )
                )

                itinerary.total_cost += selected.price

            # Lunch break

            # Afternoon activity (14:00 - 17:00)
            afternoon_activities = [
                a
                for a in suitable_activities
                if a.type
                in [ActivityType.OUTDOOR, ActivityType.CULTURAL, ActivityType.SHOPPING]
            ]
            if (
                afternoon_activities
                and len(itinerary.activities) < MAX_ACTIVITIES_PER_DAY
            ):
                selected = afternoon_activities[
                    0
                ]  # In a real implementation, we would make a smarter selection
                suitable_activities.remove(selected)

                itinerary.activities.append(
                    ScheduledActivity(
                        activity=selected,
                        date=date_str,
                        start_time="14:00",
                        end_time="17:00",
                        notes="Afternoon exploration",
                    )
                )

                itinerary.total_cost += selected.price

            # Evening activity (19:00 - 21:00)
            evening_activities = [
                a
                for a in suitable_activities
                if a.type in [ActivityType.FOOD, ActivityType.ENTERTAINMENT]
            ]
            if (
                evening_activities
                and len(itinerary.activities) < MAX_ACTIVITIES_PER_DAY
            ):
                # In a real implementation, we would make a smarter selection
                selected = evening_activities[0]

                itinerary.activities.append(
                    ScheduledActivity(
                        activity=selected,
                        date=date_str,
                        start_time="19:00",
                        end_time="21:00",
                        notes="Evening entertainment",
                    )
                )

                itinerary.total_cost += selected.price

            # Check budget constraints
            if context.budget_per_day and itinerary.total_cost > context.budget_per_day:
                # In a real implementation, we would adjust the itinerary to fit the budget
                itinerary.notes += (
                    f"\nNote: This day's activities exceed your "
                    f"daily budget of {context.budget_per_day} EUR."
                )

            # Add weather note
            if weather:
                itinerary.notes += (
                    f"\nWeather forecast: {weather.get('condition')}, "
                    f"{weather.get('temperature_celsius')}°C "
                    f"({weather.get('temperature_fahrenheit')}°F)"
                )

            # Add the itinerary to the dictionary
            itineraries[date_str] = itinerary

            # Move to the next day
            current_date += timedelta(days=1)

        return itineraries

    async def _generate_itinerary_summary(
        self, context: ActivityPlanningContext
    ) -> str:
        """
        Generate a comprehensive summary of the trip itinerary.

        Args:
            context: Activity planning context

        Returns:
            Itinerary summary text
        """
        # Prepare a specific prompt for generating a summary
        summary_prompt = (
            "Create a detailed summary of this trip itinerary to {destination} "
            "from {start_date} to {end_date} for {traveler_count} traveler(s).\n\n"
            "Please include:\n"
            "1. An overview of the activities scheduled\n"
            "2. Highlights not to be missed\n"
            "3. Budget breakdown and total cost\n"
            "4. Weather considerations\n"
            "5. Practical tips for the activities\n\n"
            "{itinerary_details}"
        )

        # Prepare itinerary details for the prompt
        itinerary_details = []
        total_cost = 0.0

        for date_str, itinerary in context.daily_itineraries.items():
            day_details = [f"Date: {date_str}"]

            if itinerary.weather_forecast:
                weather = itinerary.weather_forecast
                day_details.append(
                    f"Weather: {weather.get('condition')}, "
                    f"{weather.get('temperature_celsius')}°C"
                )

            for i, scheduled in enumerate(itinerary.activities, 1):
                activity = scheduled.activity
                day_details.append(
                    f"{i}. {activity.name} "
                    f"({scheduled.start_time} - {scheduled.end_time})\n"
                    f"   Type: {activity.type.value}, "
                    f"Price: {activity.formatted_price}\n"
                    f"   {activity.description[:100]}..."
                )

            day_details.append(f"Daily Cost: {itinerary.total_cost:.2f} EUR\n")
            total_cost += itinerary.total_cost

            itinerary_details.append("\n".join(day_details))

        messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": summary_prompt.format(
                    destination=context.destination,
                    start_date=context.start_date,
                    end_date=context.end_date,
                    traveler_count=context.traveler_count,
                    itinerary_details="\n\n".join(itinerary_details),
                ),
            },
        ]

        response = await self._call_model(messages)

        # Return the generated summary
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
