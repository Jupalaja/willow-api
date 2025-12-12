import logging
from fastapi import APIRouter, Depends, Request
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.api.chatflow.handler import handle_chatflow
from src.api.chatflow.state import ChatflowState
from src.config import settings
from src.database.db import get_db
from src.database.models import Interaction
from src.shared.schemas import (
    InteractionRequest,
    InteractionResponse,
    InteractionMessage,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chatflow", response_model=InteractionResponse)
async def handle(
    interaction_request: InteractionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Handles a user-assistant interaction for the chatflow operation,
    continuing a conversation by loading history from the database,
    appending the new message, and saving the updated history.
    """
    logger.info(f"Received chatflow request: {interaction_request.model_dump_json(indent=2)}")
    session_id = interaction_request.sessionId
    user_message = interaction_request.message

    # Find existing interaction
    result = await db.execute(
        select(Interaction).where(Interaction.session_id == session_id)
    )
    interaction = result.scalar_one_or_none()

    if interaction:
        history_messages = [
            InteractionMessage.model_validate(msg) for msg in interaction.messages
        ]
        # Get the last state from the list
        last_state = interaction.states[-1] if interaction.states else None
        current_state = ChatflowState(last_state) if last_state else ChatflowState.IDLE
        interaction_data = interaction.interaction_data or {}
        if not interaction.practice_id and interaction_request.practiceId:
            interaction.practice_id = interaction_request.practiceId
    else:
        # New interaction
        history_messages = []
        current_state = ChatflowState.IDLE
        interaction_data = {}
        interaction = Interaction(
            session_id=session_id,
            practice_id=interaction_request.practiceId,
            messages=[],
            states=[ChatflowState.IDLE.value],
            interaction_data=interaction_data,
        )
        db.add(interaction)

    # Append new user message to history
    history_messages.append(user_message)

    if interaction.practice_id:
        interaction_data["practice_id"] = interaction.practice_id

    if interaction_request.user_data:
        # Create a copy to ensure SQLAlchemy detects changes to the JSON field.
        interaction_data = interaction_data.copy()
        # If user_data already exists and is a dict, update it. Otherwise, set it.
        if "user_data" in interaction_data and isinstance(
            interaction_data["user_data"], dict
        ):
            # Also copy the nested dict to avoid modifying it in-place from the original
            interaction_data["user_data"] = interaction_data["user_data"].copy()
            interaction_data["user_data"].update(interaction_request.user_data)
        else:
            interaction_data["user_data"] = interaction_request.user_data

    logger.debug(f"Interaction data before handle_chatflow: {interaction_data}")

    openai_model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0,
    )

    sheets_service = request.app.state.sheets_service
    response_messages, new_states, tool_call, interaction_data = await handle_chatflow(
        session_id=session_id,
        history_messages=history_messages,
        current_state=current_state,
        interaction_data=interaction_data,
        model=openai_model,
        sheets_service=sheets_service,
    )

    logger.debug(f"Interaction data after handle_chatflow: {interaction_data}")

    # Update history with new messages from the handler
    history_messages.extend(response_messages)

    # Persist changes
    interaction.messages = [
        msg.model_dump(mode="json", exclude_none=True) for msg in history_messages
    ]
    # Append the new state. To ensure SQLAlchemy detects the change, we re-assign the list.
    current_states = list(interaction.states)
    for state in new_states:
        current_states.append(state.value)
        logger.info(
            f"Session {session_id}: State added: {state.value}. Full state list: {current_states}"
        )
    interaction.states = current_states
    interaction.interaction_data = interaction_data

    await db.commit()
    await db.refresh(interaction)

    logger.debug(f"Interaction data saved for session {session_id}: {interaction.interaction_data}")

    return InteractionResponse(
        sessionId=session_id,
        messages=response_messages,
        toolCall=tool_call,
        states=interaction.states,
    )