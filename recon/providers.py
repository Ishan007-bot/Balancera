"""LLM provider adapters.

SPEC section 5 allows only ``anthropic`` as an LLM dependency, and Anthropic
remains the default. This module is a documented deviation: no Anthropic key
was available during development, so the proposer was made provider-agnostic
rather than left unmeasured. The tradeoff is one optional dependency against
an unmeasured ablation row -- and the ablation row is the number that
justifies the whole architecture.

The proposer needs exactly one thing from a provider: send a system prompt
plus a user prompt, get text and a token count back. That narrow surface is
why swapping providers costs an adapter rather than a rewrite.

Every provider here is optional and imported lazily. With none installed the
pipeline still runs under ``--no-llm``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: provider name -> (default model, env var holding the key)
PROVIDERS = {
    "anthropic": ("claude-sonnet-5", "ANTHROPIC_API_KEY"),
    # Groq rotates model availability faster than most providers. `--model`
    # overrides this, and a retired id now reports what the key can use.
    "groq": ("openai/gpt-oss-120b", "GROQ_API_KEY"),
    "gemini": ("gemini-2.0-flash", "GEMINI_API_KEY"),
}

#: USD per million tokens, (input, output). Free tiers still report tokens,
#: so the cost line stays meaningful even when the bill is zero.
PRICING = {
    "claude-sonnet-5": (2.00, 10.00),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "qwen/qwen3.8-27b": (0.10, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
}


@dataclass
class Completion:
    """One model response, normalised across providers."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class Provider:
    """Base adapter. Subclasses implement :meth:`complete`."""

    name = "base"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        default_model, env_var = PROVIDERS[self.name]
        self.model = model or default_model
        self.api_key = api_key or os.environ.get(env_var)
        self._client = None

    @property
    def env_var(self) -> str:
        return PROVIDERS[self.name][1]

    def available(self) -> tuple[bool, str]:
        """(usable, why not). Checked before a run so failures are loud."""
        raise NotImplementedError

    def complete(self, system: str, user: str, temperature: float,
                 max_tokens: int) -> Completion:
        raise NotImplementedError

    def pricing(self) -> tuple[float, float]:
        return PRICING.get(self.model, (0.0, 0.0))


class AnthropicProvider(Provider):
    """The SPEC default."""

    name = "anthropic"

    def available(self):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "the anthropic package is not installed (pip install anthropic)"
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY is not set"
        return True, ""

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, system, user, temperature, max_tokens):
        response = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in response.content
                       if getattr(b, "type", "") == "text")
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            model=self.model)


class GroqProvider(Provider):
    """Groq, via its OpenAI-compatible endpoint.

    The system prompt becomes a system *message* rather than a top-level
    field -- the one structural difference from the Anthropic API that this
    adapter exists to absorb.
    """

    name = "groq"

    def available(self):
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "the openai package is not installed (pip install openai)"
        if not self.api_key:
            return False, "GROQ_API_KEY is not set"
        return True, ""

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1")
        return self._client

    def list_models(self) -> list[str]:
        """Model ids this key can actually use."""
        try:
            return sorted(m.id for m in self.client.models.list().data)
        except Exception:  # noqa: BLE001 - diagnostics must not raise
            return []

    def complete(self, system, user, temperature, max_tokens):
        try:
            response = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                # Ask for JSON at the API level as well as in the prompt. The
                # parser still strips fences defensively -- belt and braces,
                # because a malformed reply must degrade to an abstain, not a
                # crash.
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001
            # Providers retire model ids without warning. A bare
            # "model_not_found" leaves the user guessing, so name what is
            # actually available instead.
            if "model_not_found" in str(exc) or "does not exist" in str(exc):
                available = self.list_models()
                raise RuntimeError(
                    "model %r is not available on this Groq key. Available: "
                    "%s. Pass --model <id> to choose one."
                    % (self.model, ", ".join(available[:12]) or "(none listed)")
                ) from exc
            raise
        usage = getattr(response, "usage", None)
        return Completion(
            text=response.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            model=self.model)


class GeminiProvider(Provider):
    name = "gemini"

    def available(self):
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            return False, ("the google-generativeai package is not installed "
                           "(pip install google-generativeai)")
        if not self.api_key:
            return False, "GEMINI_API_KEY is not set"
        return True, ""

    @property
    def client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai
        return self._client

    def complete(self, system, user, temperature, max_tokens):
        genai = self.client
        model = genai.GenerativeModel(
            model_name=self.model, system_instruction=system)
        response = model.generate_content(
            user,
            generation_config={"temperature": temperature,
                               "max_output_tokens": max_tokens,
                               "response_mime_type": "application/json"})
        usage = getattr(response, "usage_metadata", None)
        return Completion(
            text=response.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            model=self.model)


_REGISTRY = {"anthropic": AnthropicProvider, "groq": GroqProvider,
             "gemini": GeminiProvider}


def get_provider(name: str = "anthropic", model: str | None = None,
                 api_key: str | None = None) -> Provider:
    if name not in _REGISTRY:
        raise ValueError("unknown provider %r; choose from %s"
                         % (name, ", ".join(sorted(_REGISTRY))))
    return _REGISTRY[name](model=model, api_key=api_key)


def detect_available() -> list[str]:
    """Which providers could actually run right now. Used to fail loudly."""
    return [name for name in _REGISTRY
            if get_provider(name).available()[0]]
