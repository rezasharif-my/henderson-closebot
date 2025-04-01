from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph , START , COMMAND
from typing import Annotated, Optional, Literal, List, Union
from langchain_core.runnables import RunnableLambda
from dataclasses import dataclass, field
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain.chat_models import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import AnyMessage, add_messages
from app.classes import Config
from app.handlers import clarification_node, intent_classifier_node, jack_reply_generator_node, memory_loader_node
from app.prompts import jack_system_prompt
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

    messages: Annotated[List[AnyMessage], add_messages]
    """The messages in the conversation."""

    summary: Optional[str] = None
    """Summarized user input to improve responses."""

    intent: Optional[str] = None
    """Detected intent of the user query."""

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

# Define the node to load the memory from the database
memory_loader = RunnableLambda(memory_loader_node)

# Define the node to handle the clarification question
clarification = RunnableLambda(clarification_node)

# Define the node to handle the general response
jack_reply_generator = RunnableLambda(jack_reply_generator_node)




# Define Graph
workflow = StateGraph(State, config_schema=Config)

# Nodes
workflow.add_node("intent_classifier", intent_classifier)
workflow.add_node("memory_loader", memory_loader)
workflow.add_node("clarification_node", clarification)
workflow.add_node("jack_reply_generator", handle_general_response)
workflow.add_node("faq_matcher", handle_irrelevant_response)
workflow.add_node("lead_capture_node", final_answer_router)
workflow.add_node("motivator_node", single_result_generator)
workflow.add_node("summary_node", list_result_generator)
workflow.add_node("user_save_node", list_result_generator)
workflow.add_node("end_convo_node", list_result_generator)