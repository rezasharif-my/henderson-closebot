from langchain.chat_models import ChatOpenAI
from app.db import get_recent_chat, save_user, get_user_by_thread_id, get_latest_summary
from pydantic import BaseModel, Field
from typing import Literal , Optional
from langchain_core.messages import SystemMessage
import re
llm = ChatOpenAI(model="gpt-4o", temperature=0)




# =============================================================================
# Define the intent classifier node
# =============================================================================
class DetectedIntent(BaseModel):
    intent: Literal[
        "greeting",
        "invest_interest",
        "faq",
        "lead_capture",
        "smalltalk",
        "unclear"
    ] = Field(description="User intent category")

INTENT_PROMPT = """
You are an intent classification AI. Given the user's message, classify it into one of the following intents:

- greeting: saying hi, hello, etc.
- invest_interest: user shows interest in investing or buying property
- faq: user asks about company services, offerings, pricing
- lead_capture: user shares contact info (email or phone)
- smalltalk: jokes, casual convo, not business-focused
- unclear: you can't tell the intent clearly

Respond with just one of the above keywords.

User message: "{message}"
"""

def intent_classifier_node(state):
    user_msg = state.get("user_message")
    if not user_msg:
        return {**state, "intent": "unclear"}

    prompt = INTENT_PROMPT.format(message=user_msg)
    system_msg = SystemMessage(content=prompt)

    try:
        structured_llm = llm.with_structured_output(DetectedIntent)
        result = structured_llm.invoke([system_msg])
        return {
            **state,
            "intent": result.intent
        }
    except Exception as e:
        print(f"[Intent Classifier Error] {e}")
        return {
            **state,
            "intent": "unclear"
        }

# =============================================================================
# Define the memory loader node
# =============================================================================
def memory_loader_node(state):
    """Loads user's profile and last summary into the graph state."""
    thread_id = state.get("thread_id")
    if not thread_id:
        return state

    user_info = get_user_by_thread_id(thread_id)
    summary = get_latest_summary(thread_id)

    return {
        **state,
        "summary": summary,
        "user_profile": user_info,  # Optional: if you add to your state later
    }

# =============================================================================
# Define the clarification node
# =============================================================================
class ClarificationOutput(BaseModel):
    clarification_question: str = Field(description="Follow-up question to clarify user intent")

CLARIFICATION_PROMPT = '''
You are a helpful assistant detecting ambiguity in user queries.

The user's message is unclear or open-ended. Your job is to ask a **single clarification question** to better understand their intent.

User message: "{message}"

Respond with ONLY the follow-up question.
'''

def clarification_node(state: dict) -> dict:
    user_msg = state.get("user_message")
    if not user_msg:
        return state

    prompt = CLARIFICATION_PROMPT.format(message=user_msg)
    system_msg = SystemMessage(content=prompt)

    try:
        clarifier_llm = llm.with_structured_output(ClarificationOutput)
        result = clarifier_llm.invoke([system_msg])
        return {
            **state,
            "clarification_question": result.clarification_question,
            "status": "NEED_USER_CLARIFICATION"
        }
    except Exception as e:
        print(f"[Clarification Error] {e}")
        return {
            **state,
            "clarification_question": "Can you clarify what you mean?",
            "status": "NEED_USER_CLARIFICATION"
        }
        
# =============================================================================
# Define the Jack reply generator node
# =============================================================================
llm_with_temperature = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Structured output for Jack's reply

class JackResponse(BaseModel):
    reply: str = Field(description="A reply message written in Jack Henderson's confident, motivational, and direct tone.")

# System prompt for Jack's tone
JACK_SYSTEM_PROMPT = """
You are Jack Henderson, the founder of Henderson Advocacy. Your tone is confident, direct, and motivational.

Respond to the user’s question as if YOU are Jack — no third-person narration. Keep replies punchy, confident, and helpful. Your style includes bold one-liners, direct advice, and occasional witty motivation.

Speak like a real person texting someone. Keep it sharp.

Example responses:
- "Stop renting. Start building. That’s how you create wealth."
- "You’ve got questions? I’ve got property plays."
- "Let’s stop talking and start building your portfolio."

Make sure to personalize the message where relevant, and ask a follow-up if it moves the conversation forward.
"""
# Main reply generator
def jack_reply_generator_node(state):
    messages = state.get("messages", [])
    user_summary = state.get("summary")
    user_intent = state.get("intent")

    # Compose system context with optional summary
    system_prompt = JACK_SYSTEM_PROMPT
    if user_summary:
        system_prompt += f"\n\nContext from our last chat:\n{user_summary}"

    try:
        structured_llm = llm_with_temperature.with_structured_output(JackResponse)
        response = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])

        return {
            **state,
            "answer": response.reply,
            "status": "ANSWER_GENERATED"
        }

    except Exception as e:
        print("[Jack Reply Error]", e)
        return {
            **state,
            "answer": "Sorry champ, something went wrong. Let’s try that again.",
            "status": "ANSWER_GENERATED"
        }


# =============================================================================
# Define the FAQ matcher node
# =============================================================================
# Simple FAQ matcher using hardcoded keywords and answers
# Optional improvements:
# - Replace with embedding + similarity search (ChromaDB or FAISS)
# - Replace with LangChain RAG chain using document loader
# - Add fuzzy matching or regular expression support

from langchain_core.runnables import RunnableLambda

# You could later move this into a separate file like faq_data.py
FAQ_RESPONSES = {
    "what does henderson do": "We help clients build wealth through smart property investment — no shortcuts, just strategy.",
    "how do i get started": "Simple. We start with a discovery call, then tailor a plan that fits your goals.",
    "do you work with first-time buyers": "Absolutely. First-time buyers are some of our biggest success stories.",
    "do you charge upfront": "Nope. Our success is tied to yours — we win when you win.",
    "how can i contact you": "Drop your email and phone number and we’ll reach out for a discovery call.",
}

def faq_matcher_node(state):
    user_msg = state.get("user_message", "").lower()
    for question, answer in FAQ_RESPONSES.items():
        if question in user_msg:
            return {
                **state,
                "answer": answer,
                "status": "ANSWER_GENERATED"
            }
    return state  # No match found → continue flow



# =============================================================================
# Define the lead capture node
# =============================================================================
# --- 1. Structured Output for Contact Info ---
class LeadInfo(BaseModel):
    email: Optional[str] = Field(description="User's email if shared")
    phone: Optional[str] = Field(description="User's phone number if shared")
# --- 2. Simple validators ---
def is_valid_email(email):
    return bool(re.match(r"[^@\s]+@[^@\s]+\.[a-zA-Z0-9]+$", email))

def is_valid_phone(phone):
    return bool(re.match(r"^(\+?\d{6,15})$", phone))
# --- 3. LLM Prompt for Extracting Info ---
LEAD_CAPTURE_PROMPT = """
Extract the user's email and phone number (if provided) from the message below.
Return only values, no comments or formatting.

User: "{message}"
"""

# --- 4. Main Node Function ---
def lead_capture_node(state):
    user_msg = state.get("user_message", "")
    thread_id = state.get("thread_id")

    prompt = LEAD_CAPTURE_PROMPT.format(message=user_msg)
    system_msg = SystemMessage(content=prompt)

    try:
        structured_llm = llm.with_structured_output(LeadInfo)
        result = structured_llm.invoke([system_msg])
        email = result.email if result.email and is_valid_email(result.email) else None
        phone = result.phone if result.phone and is_valid_phone(result.phone) else None

        # Optionally update state or ask again
        if email or phone:
            from app.db import save_user  # call inside to avoid circular import
            save_user(thread_id, email=email, phone=phone)

            confirmation = f"Awesome! I’ve got your info. I’ll email you at {email} and text you at {phone}. Let’s make it happen."
            return {
                **state,
                "answer": confirmation,
                "status": "ANSWER_GENERATED"
            }
        else:
            return {
                **state,
                "answer": "Mind sharing your email or number so we can follow up properly?",
                "status": "ANSWER_GENERATED"
            }

    except Exception as e:
        print(f"[Lead Capture Error] {e}")
        return {
            **state,
            "answer": "Hmm, I didn’t catch your contact info. Mind sending it again?",
            "status": "ANSWER_GENERATED"
        }
        


# =============================================================================
# Define the main motivator node
# =============================================================================
# --- 1. Motivation category classification ---
class MotivationLevel(BaseModel):
    level: Literal["high", "low", "uncertain"] = Field(
        description="Motivation level based on user's message"
    )

MOTIVATION_CHECK_PROMPT = """
You are an AI coach analyzing user motivation in this message:

"{message}"

Classify it as:
- high: very motivated or already committed
- low: hesitant, fearful, unsure, or delaying
- uncertain: unclear from the message

Only return the label.
"""

# --- 2. Motivational replies (Jack-style) ---
MOTIVATIONAL_QUOTES = [
    "You don’t need permission. You need to start.",
    "Smart people take action, not notes.",
    "If you’re waiting for perfect timing, you’ll wait forever.",
    "Regret weighs tons. Action weighs ounces.",
    "Stop making excuses. Start making moves."
]

import random

# --- 3. Main motivator node ---
def motivator_node(state):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    user_msg = state.get("user_message", "")
    if not user_msg:
        return state

    prompt = MOTIVATION_CHECK_PROMPT.format(message=user_msg)
    system_msg = SystemMessage(content=prompt)

    try:
        motivation_llm = llm.with_structured_output(MotivationLevel)
        result = motivation_llm.invoke([system_msg])

        if result.level == "low":
            quote = random.choice(MOTIVATIONAL_QUOTES)
            return {
                **state,
                "answer": quote,
                "status": "ANSWER_GENERATED"
            }
        else:
            return state  # skip if high/uncertain

    except Exception as e:
        print("[Motivator Error]", e)
        return state