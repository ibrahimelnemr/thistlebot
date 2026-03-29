from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Integration
from .wordpress import WordPressIntegration


@dataclass
class IntegrationRuntime:
    integrations: list[Integration]

    def middlewares(self) -> list[Any]:
        all_middlewares: list[Any] = []
        for integration in self.integrations:
            all_middlewares.extend(integration.workflow_middlewares())
        return all_middlewares

    def fallback_resolvers(self) -> list[Any]:
        all_resolvers: list[Any] = []
        for integration in self.integrations:
            all_resolvers.extend(integration.fallback_resolvers())
        return all_resolvers


def discover_integrations(config: dict[str, Any]) -> list[Integration]:
    # Current packs/workflows require WordPress integration tools.
    return [WordPressIntegration(config)]


def build_integration_runtime(config: dict[str, Any]) -> IntegrationRuntime:
    return IntegrationRuntime(integrations=discover_integrations(config))
