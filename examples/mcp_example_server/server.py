# MIT License
#
# Copyright (c) 2025, Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

# Initialize FastMCP server
mcp = FastMCP("UGREEN Media Server")


@mcp.tool()
def search_photos(
    keyword: Annotated[
        str,
        Field(
            default=...,
            description="""The search keyword used to find photos or images.
            This can be descriptive text or a file name.
            """,
            examples=["photos taken last August", "dog on the grass"],
        ),
    ],
) -> dict[str, Any]:
    """
    Search for photos or images.
    """
    # Simulate photo search
    result = {
        "status": "success",
        "keyword": keyword,
        "results": [
            {"filename": f"photo1_{keyword.replace(' ', '_')}.jpg", "date": "2024-08-15"},
            {"filename": f"photo2_{keyword.replace(' ', '_')}.jpg", "date": "2024-08-16"},
            {"filename": f"photo3_{keyword.replace(' ', '_')}.jpg", "date": "2024-08-17"},
        ],
        "count": 3,
    }
    print(result)
    return result


@mcp.tool(name="get_weather")
def get_weather(
    location: Annotated[
        str,
        Field(
            description="The location to get weather for.",
            examples=["Shanghai", "Beijing", "Shenzhen"],
        ),
    ],
) -> dict[str, Any]:
    """
    Retrieve a mock weather report for a given location.
    """
    current_time = datetime.datetime.now().isoformat()

    # --- Mock Logic ---
    result = {
        "status": "success",
        "location": location,
        "weather": {
            "condition": "Sunny",
            "temperature_c": 26,
            "humidity_percent": 45,
        },
        "timestamp": current_time,
    }
    return result


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(host="0.0.0.0", port=8000, path="/mcp", transport="streamable-http")
