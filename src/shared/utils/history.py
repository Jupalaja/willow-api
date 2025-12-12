from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage

from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage


def get_langchain_history(
    history_messages: list[InteractionMessage],
) -> list[BaseMessage]:
    """
    Converts the application's internal message history format to the
    format required by LangChain.

    Args:
        history_messages: A list of messages in the application's format.

    Returns:
        A list of `BaseMessage` objects ready to be sent to the model.
    """
    langchain_history = []
    for msg in history_messages:
        if msg.role == InteractionType.USER:
            langchain_history.append(HumanMessage(content=msg.message))
        elif msg.role == InteractionType.MODEL:
            langchain_history.append(AIMessage(content=msg.message))
    return langchain_history


def langchain_messages_to_interaction_messages(
    messages: list[BaseMessage],
) -> list[InteractionMessage]:
    """
    Converts a list of LangChain AIMessage or HumanMessage objects back to the
    application's internal InteractionMessage format.

    Args:
        messages: A list of `BaseMessage` objects from a model response.

    Returns:
        A list of messages in the application's internal format.
    """
    interaction_messages = []
    for message in messages:
        if isinstance(message, AIMessage):
            role = InteractionType.MODEL
            tool_calls = (
                [tc["name"] for tc in message.tool_calls] if message.tool_calls else None
            )
        elif isinstance(message, HumanMessage):
            role = InteractionType.USER
            tool_calls = None
        else:
            # Skipping other message types like ToolMessage, SystemMessage for now
            continue

        interaction_messages.append(
            InteractionMessage(
                role=role, message=str(message.content), tool_calls=tool_calls
            )
        )
    return interaction_messages
