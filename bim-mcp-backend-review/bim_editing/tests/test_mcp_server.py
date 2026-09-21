from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[2]


def test_stdio_server_lists_complete_bim_workflow():
    async def exercise():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "bim_mcp_server.py"), "--transport", "stdio"],
            cwd=ROOT,
        )
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = await session.list_tools()
                resources = await session.list_resources()
                templates = await session.list_resource_templates()
                stack = await session.read_resource("bim://engineering/stack")
                recipe = await session.read_resource(
                    "bim://authoring/recipe/assembly.window-in-wall"
                )
                references = await session.call_tool(
                    "search_ifc_reference_models",
                    {
                        "required_entities": ["IfcWall", "IfcWindow"],
                        "limit": 1,
                    },
                )
                return (
                    {tool.name for tool in tools.tools},
                    {str(resource.uri) for resource in resources.resources},
                    {
                        str(template.uriTemplate)
                        for template in templates.resourceTemplates
                    },
                    json.loads(stack.contents[0].text),
                    json.loads(recipe.contents[0].text),
                    json.loads(references.content[0].text),
                )

    (
        tool_names,
        resource_uris,
        resource_templates,
        stack,
        recipe,
        references,
    ) = anyio.run(exercise)
    assert tool_names == {
        "list_bim_recipes",
        "search_bim_recipes",
        "get_bim_recipe",
        "search_ifc_reference_models",
        "get_ifc_reference_model",
        "recover_ifc_for_editing",
        "resolve_bim_part",
        "apply_bim_revision",
        "render_bim_revision",
        "export_approved_bim_revision",
    }
    assert resource_uris == {
        "bim://engineering/stack",
        "bim://authoring/recipes",
        "bim://ifc-library/summary",
        "bim://ifc-library/relationship-patterns",
        "bim://editing/operations",
        "bim://editing/workflow",
    }
    assert resource_templates == {"bim://authoring/recipe/{recipe_id}"}
    assert stack["schema"] == "bim.engineering-stack.v1"
    assert recipe["id"] == "assembly.window-in-wall"
    assert references["count"] == 1
