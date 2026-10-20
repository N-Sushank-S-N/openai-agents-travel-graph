"""
Base agent class for the travel planner system.

This module implements the foundational Agent class that all specialized
agents in the travel planner system will inherit from. It provides common
functionality and standardized interfaces for all agents.
"""
import os
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

# Type variable for context
T = TypeVar("T")


class AgentContext(BaseModel):
    """Base class for agent context that can be passed between agents."""

    pass


class TravelPlannerAgentError(Exception):
    """Base exception for all agent-related errors."""

    pass


class InvalidConfigurationError(TravelPlannerAgentError):
    """Exception raised when agent configuration is invalid."""

    pass
# Backward compatibility for existing tests and callers
InvalidConfigurationException = InvalidConfigurationError

@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    instructions: str
    model: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_MODEL", "gemini-3.8-flash"
        )
    )
    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[Any] = field(default_factory=list)


class BaseAgent(Generic[T]):
    """
    Base class for all travel planner agents.

    This class provides the foundation for specialized agents that handle
    different aspects of travel planning, such as destination research,
    flight search, accommodation booking, etc.

    The BaseAgent implements common functionality like handling API clients,
    error management, and providing a standardized interface for all agents.
    """

    def __init__(
        self,
        config: AgentConfig,
        context_type: type[T] | None = None,
    ):
        """
        Initialize a base agent.

        Args:
            config: Configuration for the agent
            context_type: Type of context this agent handles (optional)
        """
        self.config = config
        # Use Gemini for legacy GPT model defaults when configured.
        if (
            os.getenv("GEMINI_API_KEY")
            and "generativelanguage.googleapis.com"
            in os.getenv("OPENAI_BASE_URL", "")
            and getattr(config, "model", "").startswith("gpt-")
        ):
            config.model = os.getenv("DEFAULT_MODEL", "gemini-3.8-flash")
        load_dotenv()

        gemini_key = os.getenv("GEMINI_API_KEY")

        if gemini_key:
            base_url = os.getenv(
                "OPENAI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.client = AsyncOpenAI(
                api_key=gemini_key.strip(),
                base_url=base_url.strip(),
            )
        else:
            self.client = AsyncOpenAI()
        self.context_type = context_type or AgentContext

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self.config.name

    @property
    def instructions(self) -> str:
        """Get the instructions for the agent."""
        return self.config.instructions

    def invoke(self, state: Any) -> dict[str, Any]:
        """Synchronously adapt the workflow state to the async agent interface."""

        query = getattr(getattr(state, "query", None), "raw_query", "")
        state_data = (
            state.model_dump(mode="json")
            if hasattr(state, "model_dump")
            else {}
        )

        input_text = (
            "Perform your assigned travel-planning task using the current state.\n"
            f"User request: {query}\n"
            "Current state as JSON:\n"
            f"{json.dumps(state_data, ensure_ascii=False, default=str)}\n\n"
            "Return a JSON object containing the structured results relevant to "
            "your agent's task."
        )

        async def call_agent() -> Any:
            result = await self.run(input_text)
            response = result.get("response", result) if isinstance(result, dict) else result

            if isinstance(response, dict):
                content = response.get("content")
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                return response

            return {"content": str(response)}

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(call_agent())

        # If a loop is already running, execute the coroutine in a separate thread.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(call_agent())).result()
    async def run(
        self, input_data: str | list[dict[str, Any]], context: T | None = None
    ) -> Any:
        """
        Run the agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional context for the agent

        Returns:
            Agent response or result
        """
        # This would typically integrate with the OpenAI Agents SDK
        # For now, this is a simple implementation
        raise NotImplementedError("Subclasses must implement run method")

    async def process(self, *args, **kwargs) -> Any:
        """
        Process the input according to the agent's specialized function.

        This is the main method that specialized agents will implement
        to perform their specific tasks.
        """
        raise NotImplementedError("Subclasses must implement process method")

    def _validate_config(self) -> bool:
        """Validate the agent configuration."""
        if not self.config.name:
            raise InvalidConfigurationException("Agent name cannot be empty")
        if not self.config.instructions:
            raise InvalidConfigurationException("Agent instructions cannot be empty")
        return True

    def _prepare_messages(
        self, input_data: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Prepare messages for the API call.

        Args:
            input_data: User input or conversation history

        Returns:
            List of messages formatted for the API
        """
        if isinstance(input_data, str):
            messages = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": input_data},
            ]
        # If input_data is already a list of messages, add system message if not present
        elif input_data and input_data[0].get("role") != "system":
            messages = [{"role": "system", "content": self.instructions}, *input_data]
        else:
            messages = input_data

        return messages
