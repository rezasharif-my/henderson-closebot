from langgraph.graph import END, StateGraph , START 
from typing import Annotated, Optional, Literal, List, Union , Dict
from langchain_core.runnables import RunnableLambda
from dataclasses import dataclass, field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.types import interrupt , Command
from classes import Config
from dataclasses import replace
from handlers import clarification_node, faq_matcher_node, intent_classifier_node, jack_reply_generator_node, lead_capture_node, memory_loader_node, motivator_node, summary_node, user_save_node
import sqlite3
import os


# Define the graph memory (sqlight)
DATABASE_PATH ="app/database/state_db/state_data.db"
conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

# Define the graph State
@dataclass(kw_only=True)
class State:
    """Main chatbot graph state."""
    
    thread_id: Optional[str] = "TEST"
    """Unique thread ID for the conversation."""

    messages: Annotated[List[AnyMessage], add_messages]
    """The messages in the conversation."""

    summary: Optional[str] = None
    """Summarized user input to improve responses."""

    intent: Optional[str] = None
    """Detected intent of the user query."""
    
    user_profile: Optional[Dict[str, str]] = None
    """Basic profile info like name, email, phone, platform, etc."""

    clarification_question: Optional[str] = None
    """Question for clarification."""

    status: Optional[
        Literal[
            "ANSWER_GENERATED",
            "NEED_USER_CLARIFICATION",
        ]
    ] = None
    """Status of the chat."""

    answer: Optional[Union[str, dict]] = None
    """Answer to the user query."""
# Ensure the class is available for imports
__all__ = ["State"]



# Define the graph nodes

#  Define the node to detect the intent of the user query
intent_classifier = RunnableLambda(intent_classifier_node)

# Define the node to load the memory from the database and add related data to the state
memory_loader = RunnableLambda(memory_loader_node)

# Define the node to handle the clarification question
clarification = RunnableLambda(clarification_node)

# Define the node to handle the general response (Jack Henderson's reply)
jack_reply_generator = RunnableLambda(jack_reply_generator_node)

# Define the node to handle the FAQ matching
faq_matcher = RunnableLambda(faq_matcher_node)

# Define the node to handle the lead capture
lead_capture = RunnableLambda(lead_capture_node)

# Define the node to handle the motivator
motivator = RunnableLambda(motivator_node)

# Define the node to generate the summary
summary = RunnableLambda(summary_node)

# Define the node to save the user data
user_save = RunnableLambda(user_save_node)

def clarification_handler(state: State):
    """Handle the clarification response from the user."""
    human_message = interrupt("clarification")
    state = replace(state, messages=state.messages + [human_message])
    return state

# Define the function to route the user query based on the detected intent
def route_intent(state: State) -> str:
    """Route the user query based on the detected intent."""
    if state.intent == "faq":
        return "faq_matcher"
    elif state.intent == "lead_capture":
        return "lead_capture_node"
    elif state.intent == "unclear":
        return "clarification_node"
    else:
        return "jack_reply_generator"

def route_faq(state: State) -> str:
    """Route the user query based on the detected intent."""
    answer = state.answer
    if answer:
        return "summary_node"
    else:
        return "jack_reply_generator"

# Define Graph
workflow = StateGraph(State, config_schema=Config)

# Nodes
workflow.add_node("memory_loader", memory_loader)
workflow.add_node("intent_classifier", intent_classifier)
workflow.add_node("clarification_node", clarification)
workflow.add_node("clarification_handler", clarification_handler)
workflow.add_node("jack_reply_generator", jack_reply_generator)
workflow.add_node("faq_matcher", faq_matcher)
workflow.add_node("lead_capture_node", lead_capture)
workflow.add_node("motivator_node", motivator)
workflow.add_node("summary_node", summary)
workflow.add_node("user_save_node", user_save)

# Edges
workflow.add_edge(START, "memory_loader")
workflow.add_edge("memory_loader", "intent_classifier")
workflow.add_conditional_edges(
    "intent_classifier", 
    route_intent, 
    {
        "faq_matcher": "faq_matcher",
        "lead_capture_node": "lead_capture_node",
        "clarification_node": "clarification_node",
        "jack_reply_generator": "jack_reply_generator"
    }
)
workflow.add_conditional_edges(
    "faq_matcher", 
    route_faq, 
    {
        "summary_node": "summary_node",
        "jack_reply_generator": "jack_reply_generator"
    }
)  
workflow.add_edge("lead_capture_node", "summary_node")  
workflow.add_edge("clarification_node", "clarification_handler")  
workflow.add_edge("clarification_handler", "intent_classifier")  
workflow.add_edge("jack_reply_generator", "motivator_node")  
workflow.add_edge("motivator_node", "summary_node")  
workflow.add_edge("summary_node", "user_save_node")  
workflow.add_edge("user_save_node", END)  

maingraph = workflow.compile(checkpointer=memory)

