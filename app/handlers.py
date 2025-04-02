from langchain_openai import ChatOpenAI
from utils import get_environment_ready, reset_state_fields
from db import get_recent_chat, save_chat_summary, save_user, get_user_by_thread_id, get_latest_summary
from pydantic import BaseModel, Field
from typing import Literal , Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import RemoveMessage

import re

from dataclasses import replace
get_environment_ready()

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
- lead_capture: user shares contact info (email or phone or name)
- smalltalk: jokes, casual convo, not business-focused or general questions
- *** if message starts with CLARIFIED return smalltalk ***
- unclear: you can't tell the intent clearly

Respond with just one of the above keywords.

User message: "{message}"
"""

def intent_classifier_node(state):
    user_msg = state.messages[-1].content 
    if not user_msg:
        return replace(state, intent=result.intent)

    prompt = INTENT_PROMPT.format(message=user_msg)
    system_msg = SystemMessage(content=prompt)

    try:
        structured_llm = llm.with_structured_output(DetectedIntent)
        result = structured_llm.invoke([system_msg])
        return replace(state, intent=result.intent)
    except Exception as e:
        print(f"[Intent Classifier Error] {e}")
        return replace(state, intent=result.intent)

# =============================================================================
# Define the memory loader node
# =============================================================================
def memory_loader_node(state, config):
    """Loads user's profile and last summary into the graph state from config thread_id."""
    
    state = reset_state_fields(state, ["answer", "intent", "clarification_question","status"])
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        print("[MemoryLoader] No thread_id found in config.")
        return state

    user_info = get_user_by_thread_id(thread_id)
    summary = get_latest_summary(thread_id)

    return replace(state, summary=summary, user_profile=user_info)

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
Respond as You are Jack Henderson, the founder of Henderson Advocacy. Your tone is confident, direct, and motivational.
'''

def clarification_node(state: dict) -> dict:
    user_msg = state.messages[-1].content 
    if not user_msg:
        return state

    prompt = CLARIFICATION_PROMPT.format(message=user_msg)
    system_msg = SystemMessage(content=prompt)

    try:
        clarifier_llm = llm.with_structured_output(ClarificationOutput)
        result = clarifier_llm.invoke([system_msg])
        return replace(state, clarification_question=result.clarification_question, status="NEED_USER_CLARIFICATION")
    except Exception as e:
        print(f"[Clarification Error] {e}")
        return replace(state, clarification_question="Can you clarify what you mean?", status="NEED_USER_CLARIFICATION")

        
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
    messages = state.messages
    user_summary = state.summary or ""
    user_intent = state.intent or None
    user_info = get_user_by_thread_id(state.thread_id)
    

    # Compose system context with optional summary
    system_prompt = JACK_SYSTEM_PROMPT
    if user_summary:
        system_prompt += f"\n\nContext from our last chat:\n{user_summary}"
    if user_info:
        system_prompt += f"\n\nUser info: {user_info}"

    try:
        structured_llm = llm_with_temperature.with_structured_output(JackResponse)
        response = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])
        return replace(state, answer=response.reply, status="ANSWER_GENERATED")


    except Exception as e:
        print("[Jack Reply Error]", e)
        return replace(state, answer= "Sorry champ, something went wrong. Let’s try that again.", status="ANSWER_GENERATED")


# =============================================================================
# Define the FAQ matcher node
# =============================================================================
# Simple FAQ matcher using hardcoded keywords and answers
# Optional improvements:
# - Replace with embedding + similarity search (ChromaDB or FAISS)
# - Replace with LangChain RAG chain using document loader
# - Add fuzzy matching or regular expression support


# You could later move this into a separate file like faq_data.py
FAQ_RESPONSES = {
    "what does henderson do": "We help clients build wealth through smart property investment — no shortcuts, just strategy.",
    "how do i get started": "Simple. We start with a discovery call, then tailor a plan that fits your goals.",
    "do you work with first-time buyers": "Absolutely. First-time buyers are some of our biggest success stories.",
    "do you charge upfront": "Nope. Our success is tied to yours — we win when you win.",
    "how can i contact you": "Drop your email and phone number and we’ll reach out for a discovery call.",
}

def faq_matcher_node(state):
    user_msg = state.messages[-1].content 
    for question, answer in FAQ_RESPONSES.items():
        if question in user_msg:
            return replace(state, answer=answer, status="ANSWER_GENERATED")
    return state  # No match found → continue flow



# =============================================================================
# Define the lead capture node
# =============================================================================
# --- 1. Structured Output for Contact Info ---
class LeadInfo(BaseModel):
    email: Optional[str] = Field(description="User's email if shared")
    phone: Optional[str] = Field(description="User's phone number if shared")
    name: Optional[str] = Field(description="User's name if shared")
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
    user_msg = state.messages[-1].content 
    thread_id = state.thread_id or None

    prompt = LEAD_CAPTURE_PROMPT.format(message=user_msg)
    system_msg = SystemMessage(content=prompt)

    try:
        structured_llm = llm.with_structured_output(LeadInfo)
        result = structured_llm.invoke([system_msg])
        email = result.email if result.email and is_valid_email(result.email) else None
        phone = result.phone if result.phone and is_valid_phone(result.phone) else None
        name = result.name if result.name else None

        # Optionally update state or ask again
        if name:
            from app.db import save_user 
            save_user(thread_id, name=name)
            
        if email or phone:
            from app.db import save_user 
            save_user(thread_id, email=email, phone=phone)

            confirmation = f"Awesome! I’ve got your info. I’ll contact you with the info you shared. Let’s make it happen."
            return replace(state, answer=confirmation, status="ANSWER_GENERATED")
        else:
            return replace(state, answer="Mind sharing your email or number so we can follow up properly?", status="ANSWER_GENERATED")


    except Exception as e:
        print(f"[Lead Capture Error] {e}")
        return replace(state, answer= "Hmm, I didn’t catch your contact info. Mind sending it again?", status="ANSWER_GENERATED")
        


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
    user_msg = state.messages[-1].content 
    if not user_msg:
        return state

    prompt = MOTIVATION_CHECK_PROMPT.format(message=user_msg)
    system_msg = SystemMessage(content=prompt)

    try:
        motivation_llm = llm.with_structured_output(MotivationLevel)
        result = motivation_llm.invoke([system_msg])

        if result.level == "low":
            quote = random.choice(MOTIVATIONAL_QUOTES)
            current_answer = state.answer or ""
            new_answer = current_answer.strip() + "\n\n" + quote
            return replace(state, answer=new_answer, status="ANSWER_GENERATED")
        else:
            return state

    except Exception as e:
        print("[Motivator Error]", e)
        return state


# =============================================================================
# Define the main summarizer node
# =============================================================================
# --- Main summarizer node ---
def summary_node(state):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    thread_id = state.thread_id or None
    messages = state.messages

    # Don't summarize unless there's context to work with
    if len(messages) <= 3:
        return state
    # Pull latest summary from DB
    existing_summary = get_latest_summary(thread_id) or ""

    # Summarization prompt
    if existing_summary:
        summary_prompt = (
            f"This is a summary of the user's conversation so far: {existing_summary}\n\n"
            "Based on the following chat messages, expand and refine this summary. Focus only on property goals, investment interests, or buyer concerns relevant to Henderson Advocacy. Keep it in ONE concise paragraph in natural language."
        )
    else:
        summary_prompt = (
            "Summarize the following chat messages in ONE paragraph."
            " Focus only on property goals, investment interests, or buyer concerns relevant to Henderson Advocacy. Do not mention chit-chat or unrelated info."
        )

    # Call LLM
    full_prompt = messages + [HumanMessage(content=summary_prompt)]
    response = model.invoke(full_prompt)
    new_summary = response.content.strip()

    # Save to DB
    if thread_id:
        save_chat_summary(thread_id, new_summary)

    # Trim memory to last 2 messages
    clear_old = [RemoveMessage(id=m.id) for m in messages[:-2]]
    return replace(state, summary=new_summary, messages=clear_old)


# =============================================================================
# Define the user save node
# =============================================================================
def user_save_node(state):
    thread_id = state.thread_id
    user_profile = state.user_profile or {}

    name = user_profile.get("name")
    platform = user_profile.get("platform")
    email = user_profile.get("email")
    phone = user_profile.get("phone")

    if thread_id:
        save_user(
            thread_id=thread_id,
            name=name,
            platform=platform,
            email=email,
            phone=phone,
        )

    return state  # State unchanged