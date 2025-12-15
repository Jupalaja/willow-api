from typing import Literal, Optional
from langchain_core.tools import tool


ConversationType = Literal[
    "is_question_pricing",
    "is_acknowledgment",
    "is_general_faq_question",
    "is_out_of_scope_question",
    "is_frustrated_needs_human",
    "is_bot_creation_request",
    "is_goodbye",
]


@tool
def classify_intent(intent: ConversationType) -> ConversationType:
    """Classifies the user's intent based on their message. Call this function with the most relevant classification.

    Classifications:
    - is_question_pricing: User ask questions about the price of the service, asks about subscriptions or wants to know more about the cost of building an Artificial Intelligence (AI) Chatbot.
    - is_acknowledgment: User says "thanks", "got it" or any expression for acknowledging a previous response from the agent. This does NOT include greetings like "Hi" or "Hello", or farewells like "Bye" or "Goodbye"
    - is_general_faq_question: User asks a general question about the services included, the internal workings of the agent, the AI (Artificial Intelligence) services or LLM (Large Language Model) used.
    - is_out_of_scope_question: User asks a question for which information is not available in the context. If the question cannot be answered using the provided context, use this classification.
    - is_frustrated_needs_human: User expresses frustration, wants to speak to a person, or is dissatisfied with bot responses.
    - is_bot_creation_request: User greets (e.g. "Hi", "Hello"), or expresses the desire to build a bot or says that is interested in MedbotPro's services.
    - is_goodbye: User says goodbye or indicates the conversation is over.

    Args:
        intent: The user's intent classification.
    """
    return intent


@tool
def user_accepts_book_call(user_accepts: bool) -> bool:
    """Use this tool to determine if the user accepts to book a free consultation call.

    Look for responses like:
    - "Yes, I'd like to book a call"
    - "That sounds good"
    - "Let's do it"
    - "I'm interested"
    - "How do I book?"

    Set user_accepts=True if they agree or show interest in booking.
    Set user_accepts=False if they decline, say "not right now", "maybe later", or express hesitation.

    Args:
        user_accepts: True if user wants to book discovery call, False if declining
    """
    return user_accepts


@tool
def send_book_call_link() -> str:
    """Use this tool to provide the user with the booking link for a free consultation call.

    Only call this tool AFTER user_accepts_book_call() returns True.

    The consultation call includes:
    - ~30 minutes with a developer to discuss the responses from the chatbot
    - Explanation of our approach to provide precise and human-like responses
    - Determination of next steps for building the chat assistant
    - Completely free with no obligation

    The booking link is: https://cal.com/jupalaja
    """
    return "Here's the **[Link to Book a Call](https://cal.com/jupalaja)**"


@tool
def get_user_data(name: Optional[str] = None, email: Optional[str] = None) -> dict:
    """Extracts user's name and email ONLY if explicitly provided in the message.

    If the user explicitly provides their name or email, pass them as arguments.
    If the user explicitly REFUSES to provide information (e.g. "No", "Skip"), pass name="" and email="".

    IMPORTANT:
    - Do NOT guess or hallucinate values.
    - Do NOT extract placeholders like "child's name" or "child's email".
    - Do NOT extract names of companies or other entities mentioned in the conversation context as the user's name.
    - If the user has NOT provided a name or email in this specific message, call this function with NO arguments.

    Args:
        name: The user's name if provided.
        email: The user's email if provided.
    """
    user_data = {}
    if name is not None:
        user_data["name"] = name
    if email is not None:
        user_data["email"] = email
    return user_data
