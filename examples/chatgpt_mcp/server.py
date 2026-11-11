# SPDX-License-Identifier: Apache-2.0
"""Local ChatGPT MCP proof of concept for real Cayley-Dickson multiplication."""

from typing import Annotated

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from CayleyKernel import cpu_multiply

MAX_DIMENSION = 256
Coefficient = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Vector = Annotated[
    list[Coefficient],
    Field(min_length=1, max_length=MAX_DIMENSION),
]


class Product(BaseModel):
    product: list[float]


def cayley_dickson_multiply(left: Vector, right: Vector) -> Product:
    """Compute left times right for two real Cayley-Dickson coefficient vectors.

    Supply two flat arrays of finite numbers with the same power-of-two length,
    from 1 through 256. Coefficients use CayleyKernel basis order, with the
    real coefficient first. Return the product coefficients in that order.
    """
    dimension = len(left)
    if dimension != len(right) or dimension & (dimension - 1):
        raise ValueError("Vectors must have the same power-of-two length.")

    with np.errstate(over="ignore", invalid="ignore"):
        result = cpu_multiply(left, right)
    if not np.isfinite(result).all():
        raise ValueError("Product is non-finite in float64 arithmetic.")
    return Product(product=result.tolist())


def create_server() -> FastMCP:
    server = FastMCP("CayleyKernel", host="127.0.0.1", json_response=True, stateless_http=True)
    server.add_tool(
        cayley_dickson_multiply,
        title="Multiply Cayley-Dickson vectors",
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False),
        meta={"securitySchemes": [{"type": "noauth"}]},
    )
    return server


if __name__ == "__main__":
    create_server().run(transport="streamable-http")
