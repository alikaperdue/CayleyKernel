# SPDX-License-Identifier: Apache-2.0
"""Exercise tool discovery and calls over the local Streamable HTTP app."""

import asyncio

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from examples.chatgpt_mcp.server import MAX_DIMENSION, create_server

TOOL = "cayley_dickson_multiply"
URL = "http://127.0.0.1:8000/mcp"


def exercise_tool(*, left, right):
    async def call():
        mcp = create_server()
        server_app = mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=server_app)
        async with (
            mcp.session_manager.run(),
            httpx.AsyncClient(transport=transport) as client,
            streamable_http_client(URL, http_client=client) as (reader, writer, _),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            tools = (await session.list_tools()).tools
            result = await session.call_tool(TOOL, {"left": left, "right": right})
        return tools, result

    return asyncio.run(call())


def test_octonion_product():
    # With the repository's doubling convention, e1 * e2 = e3.
    left = [0, 1, 0, 0, 0, 0, 0, 0]
    right = [0, 0, 1, 0, 0, 0, 0, 0]
    tools, result = exercise_tool(left=left, right=right)
    assert [tool.name for tool in tools] == [TOOL]
    schema = tools[0].inputSchema
    for name in ("left", "right"):
        assert schema["properties"][name]["maxItems"] == MAX_DIMENSION
        assert schema["properties"][name]["items"]["type"] == "number"
    assert not result.isError
    assert result.structuredContent == {"product": [0, 0, 0, 1, 0, 0, 0, 0]}


def test_maximum_dimension():
    left = [1] + [0] * (MAX_DIMENSION - 1)
    right = [0.5] * MAX_DIMENSION
    _, result = exercise_tool(left=left, right=right)
    assert not result.isError
    assert result.structuredContent == {"product": right}


@pytest.mark.parametrize(
    "left,right,message",
    [
        pytest.param([1, 2], [1], "same power-of-two length", id="mismatch"),
        pytest.param([1] * 3, [1] * 3, "same power-of-two length", id="non-power-of-two"),
        pytest.param([], [], "at least 1 item", id="empty"),
        pytest.param(
            [1] * (2 * MAX_DIMENSION),
            [1] * (2 * MAX_DIMENSION),
            f"at most {MAX_DIMENSION} items",
            id="oversized",
        ),
        pytest.param([[1]], [1], "valid number", id="nested"),
        pytest.param([True], [1], "valid number", id="boolean"),
        pytest.param(["1"], [1], "valid number", id="string"),
        pytest.param([None], [1], "valid number", id="null"),
        pytest.param([1e308], [1e308], "non-finite", id="product-overflow"),
    ],
)
def test_invalid_tool_call(left, right, message):
    _, result = exercise_tool(left=left, right=right)
    assert result.isError
    assert result.structuredContent is None
    assert message in result.content[0].text


@pytest.mark.parametrize("token", ["1e309", "NaN", "Infinity"])
def test_non_finite_input(token):
    async def call():
        mcp = create_server()
        transport = httpx.ASGITransport(app=mcp.streamable_http_app())
        async with mcp.session_manager.run(), httpx.AsyncClient(transport=transport) as client:
            response = await client.post(
                URL,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                content=(
                    '{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                    f'"params":{{"name":"{TOOL}","arguments":{{"left":[{token}],"right":[1]}}}}}}'
                ),
            )
        response.raise_for_status()
        return response.json()["result"]

    result = asyncio.run(call())
    assert result["isError"]
    assert "structuredContent" not in result
