from .workflows import *
from src.services.google_sheets import GoogleSheetsService
from src.shared.schemas import InteractionMessage
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Define states that should pause the conversation flow and await user input
STATES_AWAITING_USER_INPUT = {
    ChatflowState.ASK_USER_DATA,
    ChatflowState.GET_USER_DATA,
    ChatflowState.CLASSIFYING_INTENT,
    ChatflowState.VALIDATE_STATE,
    ChatflowState.AWAITING_BOOK_CALL_OFFER_RESPONSE,
    ChatflowState.AWAITING_NEW_MESSAGE,
}


async def handle_chatflow(
    session_id: str,
    history_messages: list[InteractionMessage],
    current_state: ChatflowState,
    interaction_data: Optional[dict],
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], list[ChatflowState], str | None, dict]:
    interaction_data = dict(interaction_data) if interaction_data else {}

    workflow_map = {
        ChatflowState.IDLE: idle_workflow,
        ChatflowState.CLASSIFYING_INTENT: intent_classification_workflow,
        ChatflowState.INTENT_QUESTION_CONDITION: question_condition_workflow,
        ChatflowState.PROVIDE_CONDITION_INFORMATION: provide_condition_information_workflow,
        ChatflowState.ASK_USER_DATA: ask_user_data_workflow,
        ChatflowState.INTENT_FRUSTRATED_CUSTOMER: frustrated_customer_workflow,
        ChatflowState.INTENT_OUT_OF_SCOPE_QUESTION: out_of_scope_workflow,
        ChatflowState.RECOMMENDED_DOCTOR: recommended_doctor_workflow,
        ChatflowState.REPLY_FROM_EMBEDDINGS: reply_from_embeddings_workflow,
        ChatflowState.CUSTOMER_ACKNOWLEDGES_RESPONSE: customer_acknowledges_workflow,
        ChatflowState.CONDITION_NOT_TREATED_SEND_CONTACT_INFO: condition_not_treated_workflow,
        ChatflowState.INTENT_EVENT_QUESTION: event_question_workflow,
        ChatflowState.INTENT_GENERAL_FAQ_QUESTION: general_faq_question_workflow,
        ChatflowState.INVALID_REQUEST_EMERGENCY: emergency_workflow,
        ChatflowState.VALIDATE_STATE: validate_state_workflow,
        ChatflowState.OFFER_BOOK_CALL: offer_book_call_workflow,
        ChatflowState.AWAITING_NEW_MESSAGE: await_new_message_workflow,
        ChatflowState.AWAITING_BOOK_CALL_OFFER_RESPONSE: await_book_call_response_workflow,
        ChatflowState.BOOK_CALL_OFFER_DECLINED: book_call_declined_workflow,
        ChatflowState.BOOK_CALL_OFFER_ACCEPTED: book_call_link_accepted_workflow,
        ChatflowState.INTENT_MAILING_LIST: intent_mailing_list_workflow,
        ChatflowState.INTENT_GOODBYE: goodbye_workflow,
        ChatflowState.FINAL: final_workflow,
    }

    all_new_messages = []

    next_state = current_state
    final_tool_call = None
    new_states = []

    # Loop to handle state transitions within a single turn
    for _ in range(10):  # Safety break to prevent infinite loops
        workflow_func = workflow_map.get(next_state)
        if not workflow_func:
            logger.warning(
                f"No workflow for state: {next_state}. Defaulting to intent classification."
            )
            workflow_func = intent_classification_workflow

        logger.info(
            f"Session {session_id}: Executing workflow for state {next_state}: {workflow_func.__name__}"
        )

        # The history for the tool call should include messages generated so far in this turn
        current_turn_history = history_messages + all_new_messages

        new_messages, new_state, tool_call, interaction_data = await workflow_func(
            current_turn_history, interaction_data, model, sheets_service
        )

        if new_messages:
            all_new_messages.extend(new_messages)
        if tool_call:
            final_tool_call = tool_call

        if new_state == next_state:
            # State is stable, break loop
            break

        new_states.append(new_state)
        next_state = new_state

        if (new_messages or tool_call) and next_state in STATES_AWAITING_USER_INPUT:
            # If workflow produced output for the user and requires user input, stop for this turn
            break

    return all_new_messages, new_states, final_tool_call, interaction_data
