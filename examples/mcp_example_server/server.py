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
from typing import Annotated, Any, Literal

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

MediaFeature = Literal[
    "music",
    "video",
    "unknown"
]
@mcp.tool(name="play_media")
def play_media(
    media_type: Annotated[
        MediaFeature,
        Field(
            description="Type of media to play (music, video, unknown). If don't know which type, use unknown.",
            examples=["music", "video", "unknown"],
        ),
    ],
    title: Annotated[
        str,
        Field(
            description="Title or identifier of the media item.",
            examples=["Lo-fi beats", "Company all hands", "Love Story"],
        ),
    ],
) -> dict[str, Any]:
    """Simulate playing media on a target device."""

    result = {
        "status": "playing",
        "media_type": media_type,
        "title": title,
    }
    print(result)
    return result


@mcp.tool(name="control_device")
def control_device(
    device: Annotated[
        str,
        Field(
            description="Device identifier to control.",
            examples=["air_purifier", "smart_plug_1"],
        ),
    ],
    action: Annotated[
        str,
        Field(
            description="Action to perform (power_on, power_off, set_mode).",
            examples=["power_on", "power_off", "set_mode"],
        ),
    ],
    mode: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional mode or setting to apply.",
            examples=["eco", "sleep"],
        ),
    ] = None,
) -> dict[str, Any]:
    """Simulate basic device control commands."""

    result = {
        "status": "success",
        "device": device,
        "action": action,
        "mode": mode,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    print(result)
    return result


# Define the direction of the brightness change
BrightnessAction = Literal["increase", "decrease", "set"]


@mcp.tool(name="adjust_led_brightness")
def adjust_led_brightness(
    action: Annotated[
        BrightnessAction,
        Field(
            description=(
                "The action to perform: 'increase' (add to current), 'decrease' (subtract from current), "
                "or 'set' (set to absolute value)."
            )
        ),
    ],
    value: Annotated[
        int,
        Field(
            default=25,
            description=(
                "The value associated with the action. "
                "If action is 'set', this is the target brightness (0-100). "
                "If action is 'increase'/'decrease', this is the relative amount to change."
            ),
            ge=0,  # Greater than or equal to 0 (Pydantic constraint)
            le=100,  # Less than or equal to 100 (Pydantic constraint)
        ),
    ],
) -> dict[str, Any]:
    """
    Controls the LED light brightness. Can set a specific percentage or adjust relatively.

    Use this tool when the user says:
    - "Set brightness to 80" (action='set', value=80)
    - "Make the light brighter" (action='increase', value=25)
    - "Dim the light by 10" (action='decrease', value=10)
    """

    # --- Mock Logic ---
    current_time = datetime.datetime.now().isoformat()
    # assume current brightness (for demo logic only)
    mock_current_brightness = 50
    final_brightness = 0

    # Calculate final brightness based on action
    if action == "set":
        final_brightness = value
    elif action == "increase":
        final_brightness = min(mock_current_brightness + value, 100)
    elif action == "decrease":
        final_brightness = max(mock_current_brightness - value, 0)

    return {
        "status": "Success",
        "message": f"LED brightness successfully updated. Action: '{action}', \
            Value: {value}. Resulting Level: {final_brightness}%",
        "data": {"action": action, "input_value": value, "final_brightness": final_brightness},
        "timestamp": current_time,
    }



if __name__ == "__main__":
    # Run the MCP server
    mcp.run(host="0.0.0.0", port=8000, path="/mcp", transport="streamable-http")
