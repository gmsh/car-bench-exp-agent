"""
Tool call type definitions for CAR-bench agents.

Defines Pydantic models for tool call data structures used in A2A DataPart content.
Agents should return proper A2A Messages with TextPart (for natural language) and 
DataPart (for structured tool calls).

Example usage:
    from a2a.types import Message, Part, TextPart, DataPart, Role
    from tool_call_types import ToolCall, ToolCallsData
    
    # Create tool call response with optional text
    parts = []
    if content_text:
        parts.append(Part(root=TextPart(kind="text", text=content_text)))
    
    tool_calls_data = ToolCallsData(
        tool_calls=[
            ToolCall(tool_name="get_current_location", arguments={})
        ]
    )
    parts.append(Part(root=DataPart(
        kind="data",
        data={"tool_calls": tool_calls_data.model_dump()}
    )))
    
    message = Message(
        role=Role.agent,
        parts=parts,
        message_id="...",
        context_id="..."
    )
"""

import json
from typing import Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """
    A single tool call with name and arguments.
    This is the data structure embedded in A2A DataPart.
    """

    tool_name: str = Field(description="The name of the tool to call.")
    arguments: dict = Field(description="The arguments to pass to the tool.")

    def __str__(self) -> str:
        return f"ToolCall(tool_name={self.tool_name}, arguments={json.dumps(self.arguments)})"


class ToolCallsData(BaseModel):
    """
    Data structure for tool calls, to be embedded in A2A DataPart.
    This represents the machine-readable intent to call tools.
    """

    tool_calls: list[ToolCall] = Field(
        description="List of tool calls to execute."
    )

    def __str__(self) -> str:
        calls_str = ", ".join(str(tc) for tc in self.tool_calls)
        return f"ToolCallsData([{calls_str}])"
