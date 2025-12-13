import logging

from langchain_core.language_models import BaseChatModel

from .state import ChatflowState
from .knowledge_data import *
from .prompts import *
from .tools import *
from src.services.embeddings import retrieve_data
from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage
from src.shared.utils.functions import (call_single_tool,generate_response_text)
from src.shared.utils.history import get_langchain_history

logger = logging.getLogger(__name__)


STATES_TO_SKIP_USER_DATA=[
    ChatflowState.INTENT_OUT_OF_SCOPE_QUESTION,
    ChatflowState.INTENT_GENERAL_FAQ_QUESTION,
]

async def _send_message(
    _history_messages: list[InteractionMessage],
    _model: BaseChatModel,
    message: str,
    next_state: ChatflowState,
    interaction_data: dict,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    response_messages = [InteractionMessage(role=InteractionType.MODEL, message=message)]
    return response_messages, next_state, None, interaction_data


async def idle_workflow(
    _history_messages: list[InteractionMessage],
    interaction_data: dict,
    _model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return (
        [],
        ChatflowState.CLASSIFYING_INTENT,
        None,
        interaction_data,
    )


async def intent_classification_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    practice_id = interaction_data.get("practice_id")
    if practice_id and history_messages:
        query = history_messages[-1].message
        response, found = retrieve_data(query=query, practice_id=practice_id)
        if found:
            interaction_data["embeddings_response"] = response
            return [], ChatflowState.REPLY_FROM_EMBEDDINGS, None, interaction_data
        else:
            interaction_data.pop("embeddings_response", None)

    langchain_messages = get_langchain_history(history_messages)
    context = f"## FAQ Information\n{FAQ_DATA}"
    tool_results = await call_single_tool(
        langchain_messages, model, classify_intent, CHATFLOW_SYSTEM_PROMPT, context
    )
    intent = tool_results.get("classify_intent")

    state_map = {
        "is_question_pricing": ChatflowState.INTENT_QUESTION_PRICING,
        "is_general_faq_question": ChatflowState.INTENT_GENERAL_FAQ_QUESTION,
        "is_out_of_scope_question": ChatflowState.INTENT_OUT_OF_SCOPE_QUESTION,
        "is_frustrated_needs_human": ChatflowState.INTENT_FRUSTRATED_CUSTOMER,
        "is_acknowledgment": ChatflowState.CUSTOMER_ACKNOWLEDGES_RESPONSE,
        "is_goodbye": ChatflowState.INTENT_GOODBYE,
        "is_mailing_list": ChatflowState.INTENT_MAILING_LIST,
        "is_bot_creation_request": ChatflowState.INTENT_QUESTION_BOT_CREATION,
    }
    next_state = state_map.get(
        intent, ChatflowState.CLASSIFYING_INTENT
    )  # Fallback to re-classify

    # Check if we need to capture user data
    if next_state not in STATES_TO_SKIP_USER_DATA:
        user_data = interaction_data.get("user_data", {})
        data_refused = interaction_data.get("data_refused", False)
        # If email is missing and user hasn't refused data collection, ask for it
        if not user_data.get("email") and not data_refused:
            return [], ChatflowState.ASK_USER_DATA, None, interaction_data

    return [], next_state, None, interaction_data


async def ask_user_data_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, get_user_data, CHATFLOW_SYSTEM_PROMPT
    )
    extracted_data = tool_results.get("get_user_data")

    if extracted_data:
        # Merge extracted data
        current_user_data = interaction_data.get("user_data", {})
        if "name" in extracted_data:
            current_user_data["name"] = extracted_data["name"]
        if "email" in extracted_data:
            current_user_data["email"] = extracted_data["email"]
        interaction_data["user_data"] = current_user_data

        # Determine history for classification
        # We want to re-evaluate the intent of the message that triggered the data request.
        # So we exclude the last message (User providing data) and the one before (Model requesting data).
        classification_history = history_messages
        if len(history_messages) >= 3:
            classification_history = history_messages[:-2]

        # Check for refusal
        if extracted_data.get("name") == "" or extracted_data.get("email") == "":
            interaction_data["data_refused"] = True
            # Proceed to intent classification as if data collection is done (skipped)
            return await intent_classification_workflow(
                classification_history, interaction_data, model
            )

        # If we got data, we proceed to intent classification
        new_messages, new_state, tool_call, interaction_data = await intent_classification_workflow(
            classification_history, interaction_data, model
        )

        if new_state == ChatflowState.ASK_USER_DATA:
             # Logic to prevent infinite loop if we just got data but classification sends us back to ASK_USER_DATA
             return await _send_message(
                history_messages,
                model,
                PROMPT_ASK_USER_DATA,
                ChatflowState.ASK_USER_DATA,
                interaction_data,
            )

        return new_messages, new_state, tool_call, interaction_data

    # If no data extracted and no refusal, we ask the user
    response_text = await generate_response_text(
        history_messages,
        model,
        CHATFLOW_SYSTEM_PROMPT,
        context=INSTRUCTION_ACKNOWLEDGE_AND_ASK_USER_DATA,
    )

    if not response_text:
        return await _send_message(
            history_messages,
            model,
            PROMPT_ASK_USER_DATA,
            ChatflowState.ASK_USER_DATA,
            interaction_data,
        )

    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.ASK_USER_DATA,
        None,
        interaction_data,
    )


async def frustrated_customer_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_FRUSTRATED_CUSTOMER_OFFER_BOOK_CALL,
        ChatflowState.OFFER_BOOK_CALL,
        interaction_data,
    )


async def out_of_scope_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    context = f"{PROMPT_OUT_OF_SCOPE_QUESTION}\n\n{FAQ_DATA}"
    response_text = await generate_response_text(
        history_messages,
        model,
        CHATFLOW_SYSTEM_PROMPT,
        context=context,
    )
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.OFFER_BOOK_CALL,
        None,
        interaction_data,
    )


async def reply_from_embeddings_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    embeddings_response = interaction_data.get("embeddings_response", "")
    return await _send_message(
        history_messages,
        model,
        embeddings_response,
        ChatflowState.OFFER_BOOK_CALL,
        interaction_data,
    )


async def customer_acknowledges_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        ACKNOWLEDGMENT_MESSAGE,
        ChatflowState.AWAITING_NEW_MESSAGE,
        interaction_data,
    )


async def general_faq_question_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    instruction = (
        "Answer the user's question based on the provided context. "
        f"If the answer is not found in the context, respond with the following message: '{OUTPUT_MESSAGE_ADVANCED_MEDICAL_QUESTION}'"
    )
    context = f"{instruction}\n\n{FAQ_DATA}"
    response_text = await generate_response_text(
        history_messages,
        model,
        CHATFLOW_SYSTEM_PROMPT,
        context=context,
    )
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.AWAITING_NEW_MESSAGE,
        None,
        interaction_data,
    )


async def offer_book_call_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_OFFER_BOOK_CALL,
        ChatflowState.AWAITING_BOOK_CALL_OFFER_RESPONSE,
        interaction_data,
    )


async def await_new_message_workflow(
    _history_messages: list[InteractionMessage],
    interaction_data: dict,
    _model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    # This state loops back to intent classification for a new user query
    return [], ChatflowState.CLASSIFYING_INTENT, None, interaction_data


async def await_book_call_response_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, user_accepts_book_call, CHATFLOW_SYSTEM_PROMPT
    )
    accepts = tool_results.get("user_accepts_book_call", False)
    next_state = (
        ChatflowState.BOOK_CALL_OFFER_ACCEPTED
        if accepts
        else ChatflowState.BOOK_CALL_OFFER_DECLINED
    )
    return [], next_state, None, interaction_data


async def book_call_declined_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    full_message = f"{PROMPT_PROVIDE_CONTACT_INFO}\n\n{PROMPT_OFFER_NEWSLETTER}"
    return await _send_message(
        history_messages,
        model,
        full_message,
        ChatflowState.AWAITING_NEW_MESSAGE,
        interaction_data,
    )


async def book_call_link_accepted_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, send_book_call_link, CHATFLOW_SYSTEM_PROMPT
    )
    # The send_book_call_link tool returns the message to send
    booking_link_text = tool_results.get("send_book_call_link")

    if booking_link_text:
        interaction_data["sent_book_call_link"] = True

    # Retrieve accumulated data from previous states
    bot_creation_resp = interaction_data.get("bot_creation_response")
    account_creation_resp = interaction_data.get("account_creation_response")
    pricing_resp = interaction_data.get("pricing_response")

    # Generate a friendly response including the link and the newsletter offer
    context_parts = ["Create a natural, cohesive response that includes:"]

    if bot_creation_resp:
        context_parts.append(f"- This acknowledgment: {bot_creation_resp}")

    if pricing_resp:
        context_parts.append(f"- This pricing info: {pricing_resp}")

    if account_creation_resp:
        context_parts.append(f"- This account creation info: {account_creation_resp}")

    if booking_link_text:
        context_parts.append(f"- This booking information: {booking_link_text}")

    context_parts.append(f"- This newsletter offer: {PROMPT_OFFER_NEWSLETTER}")
    context_parts.append(
        "\nCreate a single, flowing response. Just provide a welcoming message before the booking link and newsletter offer, integrating all parts naturally.")
    context = "\n".join(context_parts)

    full_message = await generate_response_text(
        history_messages,
        model,
        system_prompt=CHATFLOW_SYSTEM_PROMPT,
        context=context,
    )

    if not full_message:
        # Fallback if generation fails
        message_parts = []
        if bot_creation_resp:
            message_parts.append(bot_creation_resp)
        if pricing_resp:
            message_parts.append(pricing_resp)
        if account_creation_resp:
            message_parts.append(account_creation_resp)
        if not message_parts:
            message_parts.append("Great! I can help with that.")
        if booking_link_text:
            message_parts.append(booking_link_text)
        message_parts.append(PROMPT_OFFER_NEWSLETTER)
        full_message = "\n\n".join(message_parts)

    # Cleanup used interaction data
    interaction_data.pop("bot_creation_response", None)
    interaction_data.pop("account_creation_response", None)
    interaction_data.pop("pricing_response", None)

    return await _send_message(
        history_messages,
        model,
        full_message,
        ChatflowState.AWAITING_NEW_MESSAGE,
        interaction_data,
    )


async def intent_mailing_list_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    await call_single_tool(
        langchain_messages, model, save_to_mailing_list, CHATFLOW_SYSTEM_PROMPT
    )

    full_message = PROMPT_ADDED_TO_MAILING_LIST

    response_message = InteractionMessage(
        role=InteractionType.MODEL, message=full_message
    )
    full_conversation = history_messages + [response_message]

    return (
        [response_message],
        ChatflowState.AWAITING_NEW_MESSAGE,
        None,
        interaction_data,
    )


async def final_workflow(
    _history_messages: list[InteractionMessage],
    interaction_data: dict,
    _model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    # This state is terminal, it does not produce any message and keeps the same state.
    return [], ChatflowState.FINAL, None, interaction_data


async def goodbye_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_INTENT_GOODBYE,
        ChatflowState.FINAL,
        interaction_data,
    )


async def intent_question_bot_creation_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    response_text = await generate_response_text(
        history_messages,
        model,
        CHATFLOW_SYSTEM_PROMPT,
        context="The user is interested in creating a bot. Provide a very short, friendly, and enthusiastic response acknowledging their interest.",
    )
    interaction_data["bot_creation_response"] = response_text
    return (
        [],
        ChatflowState.INVITE_CREATE_ACCOUNT,
        None,
        interaction_data,
    )


async def invite_create_account_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    interaction_data["account_creation_response"] = PROMPT_INVITE_CREATE_ACCOUNT
    return (
        [],
        ChatflowState.BOOK_CALL_OFFER_ACCEPTED,
        None,
        interaction_data,
    )


async def intent_question_pricing_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    interaction_data["pricing_response"] = PROMPT_INTENT_QUESTION_PRICING
    return (
        [],
        ChatflowState.BOOK_CALL_OFFER_ACCEPTED,
        None,
        interaction_data,
    )
