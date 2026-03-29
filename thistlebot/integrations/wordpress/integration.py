from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...core.tools.base import ToolEntry, ToolSpec
from ..base import Integration
from .fallback import BlogDraftFallbackResolver
from .middleware import WordPressPublishGuard
from .tools import WordPressTools

if TYPE_CHECKING:
    from ...core.tools.registry import ToolRegistry


class WordPressIntegration(Integration):
    name = "wordpress"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.tools = WordPressTools(config)

    def register_tools(self, registry: "ToolRegistry") -> None:
        registry.register(
            ToolEntry(
                spec=ToolSpec(
                    name="wordpress.list_sites",
                    description="List WordPress.com sites available to the configured REST token.",
                    input_schema={"type": "object", "properties": {}},
                ),
                execute=self.tools.list_sites,
                source="integration:wordpress",
            )
        )
        registry.register(
            ToolEntry(
                spec=ToolSpec(
                    name="wordpress.list_posts",
                    description="List posts for a WordPress site via REST API.",
                    input_schema={
                        "type": "object",
                        "required": ["site"],
                        "properties": {
                            "site": {"type": "string"},
                            "number": {"type": "integer"},
                            "status": {"type": "string"},
                        },
                    },
                ),
                execute=self.tools.list_posts,
                source="integration:wordpress",
            )
        )
        registry.register(
            ToolEntry(
                spec=ToolSpec(
                    name="wordpress.create_post",
                    description="Create a WordPress post via REST API.",
                    input_schema={
                        "type": "object",
                        "required": ["site", "title", "content"],
                        "properties": {
                            "site": {"type": "string"},
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                            "tags": {
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                ]
                            },
                            "categories": {
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                ]
                            },
                        },
                    },
                    risk_level="medium",
                ),
                execute=self.tools.create_post,
                source="integration:wordpress",
            )
        )
        registry.register(
            ToolEntry(
                spec=ToolSpec(
                    name="wordpress.update_post",
                    description="Update a WordPress post via REST API.",
                    input_schema={
                        "type": "object",
                        "required": ["site", "post_id"],
                        "properties": {
                            "site": {"type": "string"},
                            "post_id": {"type": "integer"},
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    },
                    risk_level="medium",
                ),
                execute=self.tools.update_post,
                source="integration:wordpress",
            )
        )
        registry.register(
            ToolEntry(
                spec=ToolSpec(
                    name="wordpress.get_post",
                    description="Get a WordPress post by ID via REST API.",
                    input_schema={
                        "type": "object",
                        "required": ["site", "post_id"],
                        "properties": {
                            "site": {"type": "string"},
                            "post_id": {"type": "integer"},
                        },
                    },
                ),
                execute=self.tools.get_post,
                source="integration:wordpress",
            )
        )

    def workflow_middlewares(self) -> list[WordPressPublishGuard]:
        return [WordPressPublishGuard()]

    def fallback_resolvers(self) -> list[BlogDraftFallbackResolver]:
        return [BlogDraftFallbackResolver()]
