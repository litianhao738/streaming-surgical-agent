"""Closed OpenRouter routing profiles bound to canonical API requests."""

from __future__ import annotations

from collections.abc import Mapping

from surgical_agent.api.errors import ApiContractError

OPENROUTER_ROUTING_PAYLOAD_KEY = "openrouter_routing_profile"
STRICT_OPENAI_ROUTING_PROFILE = "strict_openai"
STRICT_GOOGLE_AI_STUDIO_ROUTING_PROFILE = "strict_google_ai_studio"
STRICT_ANTHROPIC_ROUTING_PROFILE = "strict_anthropic"
STRICT_XAI_ROUTING_PROFILE = "strict_xai"
STRICT_ALIBABA_ROUTING_PROFILE = "strict_alibaba"
LATENCY_FALLBACK_ROUTING_PROFILE = "latency_fallback"
OPENROUTER_ROUTING_PROFILES = frozenset(
    {
        STRICT_OPENAI_ROUTING_PROFILE,
        STRICT_GOOGLE_AI_STUDIO_ROUTING_PROFILE,
        STRICT_ANTHROPIC_ROUTING_PROFILE,
        STRICT_XAI_ROUTING_PROFILE,
        STRICT_ALIBABA_ROUTING_PROFILE,
        LATENCY_FALLBACK_ROUTING_PROFILE,
    }
)


def routing_profile_from_options(options: Mapping[str, object]) -> str:
    """Return a validated profile, preserving strict routing as the safe default."""

    if not isinstance(options, Mapping):
        raise TypeError("OpenRouter provider options must be a mapping")
    value = options.get("routing_profile", STRICT_OPENAI_ROUTING_PROFILE)
    if not isinstance(value, str) or value not in OPENROUTER_ROUTING_PROFILES:
        raise ApiContractError("OpenRouter routing_profile is unsupported")
    return value


def routing_profile_from_request_payload(payload: Mapping[str, object]) -> str:
    """Read the cache-bound request profile with strict legacy compatibility."""

    if not isinstance(payload, Mapping):
        raise TypeError("OpenRouter request payload must be a mapping")
    value = payload.get(
        OPENROUTER_ROUTING_PAYLOAD_KEY,
        STRICT_OPENAI_ROUTING_PROFILE,
    )
    if not isinstance(value, str) or value not in OPENROUTER_ROUTING_PROFILES:
        raise ApiContractError("OpenRouter request routing profile is unsupported")
    return value


def provider_preferences(profile: str) -> dict[str, object]:
    """Translate one closed profile into the exact OpenRouter provider object."""

    if profile == STRICT_OPENAI_ROUTING_PROFILE:
        return {
            "only": ["openai"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    if profile == STRICT_GOOGLE_AI_STUDIO_ROUTING_PROFILE:
        return {
            "only": ["google-ai-studio"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    if profile == STRICT_ANTHROPIC_ROUTING_PROFILE:
        return {
            "only": ["anthropic"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    if profile == STRICT_XAI_ROUTING_PROFILE:
        return {
            "only": ["xai"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    if profile == STRICT_ALIBABA_ROUTING_PROFILE:
        return {
            "only": ["alibaba"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    if profile == LATENCY_FALLBACK_ROUTING_PROFILE:
        return {
            "only": ["openai", "azure", "amazon-bedrock"],
            "allow_fallbacks": True,
            "require_parameters": True,
            "sort": "latency",
        }
    raise ApiContractError("OpenRouter routing profile is unsupported")


def request_routing_payload(
    *,
    provider: str,
    provider_options: Mapping[str, object],
) -> dict[str, str]:
    """Return cache-bound routing metadata only for OpenRouter requests."""

    if provider != "openrouter":
        return {}
    return {
        OPENROUTER_ROUTING_PAYLOAD_KEY: routing_profile_from_options(provider_options)
    }
