"""LLM engine for research and analysis."""

import json
import logging
from typing import Any

from pydantic import BaseModel

from trading_bot.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Response from LLM analysis."""

    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int

    def as_json(self) -> dict[str, Any] | None:
        """Try to parse content as JSON."""
        try:
            # Try to extract JSON from markdown code blocks
            content = self.content
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None


class LLMEngine:
    """Engine for running LLM-based research and analysis."""

    def __init__(self) -> None:
        """Initialize the LLM engine."""
        self.settings = get_settings()
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate LLM client."""
        if self.settings.llm_provider == LLMProvider.ANTHROPIC:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key.get_secret_value()
            )
            logger.info(f"LLM engine initialized with Anthropic ({self.settings.anthropic_model})")
        else:
            import openai

            self._client = openai.OpenAI(
                api_key=self.settings.openai_api_key.get_secret_value()
            )
            logger.info(f"LLM engine initialized with OpenAI ({self.settings.openai_model})")

    def analyze(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Run analysis using the configured LLM.

        Args:
            prompt: The user prompt/query
            system_prompt: Optional system prompt for context
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)

        Returns:
            LLMResponse with the analysis results
        """
        if self.settings.llm_provider == LLMProvider.ANTHROPIC:
            return self._analyze_anthropic(prompt, system_prompt, max_tokens, temperature)
        else:
            return self._analyze_openai(prompt, system_prompt, max_tokens, temperature)

    def _analyze_anthropic(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Run analysis using Anthropic Claude."""
        messages = [{"role": "user", "content": prompt}]

        kwargs: dict[str, Any] = {
            "model": self.settings.anthropic_model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _analyze_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Run analysis using OpenAI GPT."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            provider="openai",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def structured_output(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any] | None:
        """Get structured JSON output from the LLM.

        Uses lower temperature for more consistent output.
        """
        full_prompt = f"{prompt}\n\nRespond ONLY with valid JSON, no other text."

        response = self.analyze(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        return response.as_json()

    def multi_turn_analysis(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Run multi-turn analysis with conversation history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            LLMResponse with the latest response
        """
        if self.settings.llm_provider == LLMProvider.ANTHROPIC:
            kwargs: dict[str, Any] = {
                "model": self.settings.anthropic_model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self._client.messages.create(**kwargs)

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                provider="anthropic",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        else:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)

            response = self._client.chat.completions.create(
                model=self.settings.openai_model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                provider="openai",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
