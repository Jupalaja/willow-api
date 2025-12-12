import datetime
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool

from src.config import settings
from src.services.google_sheets import GoogleSheetsService
from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage
from src.shared.utils.history import get_langchain_history

logger = logging.getLogger(__name__)


async def call_single_tool(
    messages: List[BaseMessage],
    model: BaseChatModel,
    tool_instance: BaseTool,
    system_prompt: str,
    context: str | None = None,
) -> Dict[str, Any]:
    """
    Calls a single tool with the given messages and model.

    This function takes a list of messages, a chat model, a tool function,
    and a system prompt. It binds the tool to the model, invokes the model
    with the messages, and if the model decides to call the tool, it executes
    the tool with the provided arguments and returns the result.
    """
    model_with_tools = model.bind_tools(
        [tool_instance],
        tool_choice=tool_instance.name
    )

    full_system_prompt = system_prompt
    if context:
        full_system_prompt += f"\n\n## Context\n{context}"
    prompt_messages = [SystemMessage(content=full_system_prompt)] + messages

    try:
        ai_msg = await model_with_tools.ainvoke(prompt_messages)

        if not isinstance(ai_msg, AIMessage):
            logger.warning(f"Expected an AIMessage, but got {type(ai_msg).__name__}")
            return {}

        if not ai_msg.tool_calls:
            logger.warning("Model did not call a tool.")
            return {}

        tool_call = ai_msg.tool_calls[0]
        logger.info(
            f"Calling tool: {tool_call['name']} with args: {tool_call['args']}"
        )

        tool_output = tool_instance.invoke(tool_call["args"])

        return {tool_call["name"]: tool_output}
    except Exception as e:
        logger.error(f"Error in call_single_tool: {e}", exc_info=True)
        return {}


async def generate_response_text(
    history_messages: list[InteractionMessage],
    model: BaseChatModel,
    system_prompt: str,
    context: str | None = None,
) -> str:
    """
    Generate a response text without any tool calls.

    Args:
        history_messages: The conversation history
        model: The LangChain chat model
        system_prompt: The system prompt
        context: Optional context to append to system prompt

    Returns:
        The generated response text
    """
    full_system_prompt = system_prompt
    if context:
        full_system_prompt += f"\n\n## Context\n{context}"

    langchain_messages = [
        SystemMessage(content=full_system_prompt)
    ] + get_langchain_history(history_messages)

    try:
        response = await model.ainvoke(langchain_messages)
        return str(response.content)
    except Exception as e:
        logger.error(f"Error in generate_response_text: {e}")
        return ""

async def write_candidato_a_empleo_to_sheet(
    interaction_data: dict,
    conversation: List[InteractionMessage],
    sheets_service: Optional[GoogleSheetsService]
):
    if interaction_data.get("sheet_row_added"):
        logger.info("Data for this conversation was added to sheet. Skipping write.")
        return

    if not settings.GOOGLE_SHEET_ID_EXPORT or not sheets_service:
        logger.warning(
            "Spreadsheet ID for export not configured or sheets service not available. Skipping write."
        )
        return

    try:
        worksheet = sheets_service.get_worksheet(
            spreadsheet_id=settings.GOOGLE_SHEET_ID_EXPORT,
            worksheet_name="TESTS",
        )
        if not worksheet:
            logger.error("Could not find TESTS worksheet.")
            return

        date_and_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
        user_data = interaction_data.get("user_data") or {}
        name = user_data.get("name")
        email = user_data.get("email")

        conversation_lines = []
        for msg in conversation:
            if msg.role == InteractionType.USER:
                conversation_lines.append(f"User: {msg.message}")
            elif msg.role == InteractionType.MODEL:
                conversation_lines.append(f"Linden: {msg.message}")
        conversation_str = "\n".join(conversation_lines)

        row_to_append = [
            date_and_time,
            name,
            email,
            conversation_str,
        ]

        sheets_service.append_row(worksheet, row_to_append)
        interaction_data["sheet_row_added"] = True
        logger.info("Successfully wrote data for job candidate to Google Sheet and marked as added.")

    except Exception as e:
        logger.error(f"Failed to write to Google Sheet: {e}", exc_info=True)
