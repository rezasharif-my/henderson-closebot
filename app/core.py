import uuid
from typing import Any, Dict
from classes import Config
from graph import maingraph
from langchain_core.messages import HumanMessage
from langgraph.types import Command


def generate_answer(message: str, thread_id: str = None, platform: str = "web") -> Dict[str, Any]:
    """
    Generate an answer from the chatbot based on the user query.
    Automatically handles thread ID and wraps message in HumanMessage format.
    """

    # Generate thread_id if not provided
    if not thread_id:
        thread_id = f"{platform}_{str(uuid.uuid4())[:8]}"

    # Build state input
    input_state = {
        "messages": [HumanMessage(content=message)],
        "thread_id": thread_id,
        "user_profile": {"platform": platform}
    }

    # Setup config
    config: Config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    last_state = maingraph.get_state(config)
    # Check if clarification was handled
    if last_state.values.get("status") == "NEED_USER_CLARIFICATION":
        
        new_message = f"""
        CLARIFIED 
        /// LAST USER QUESTION : {last_state.values.get("messages")[-1].content}  ///
        /// AI CLARIFICATION QUESTION : {last_state.values.get("clarification_question")}  ///
        /// USER CLARIFICATION ANSWER : {message}  ///
        """
        # Resume the graph
        output = maingraph.invoke(Command(resume=new_message), config)
    else:
        # Invoke the graph
        output = maingraph.invoke(input_state, config)
    final_state = maingraph.get_state(config)
    if final_state.values.get("clarification_question") is not None and final_state.values.get("status") == "NEED_USER_CLARIFICATION":
        return {
            "thread_id": thread_id,
            "answer": final_state.values.get("clarification_question"),
            "status": final_state.values.get("status"),
            "user_profile": final_state.values.get("user_profile"),
        }

    return {
        "thread_id": thread_id,
        "intent": final_state.values.get("intent"),
        "answer": final_state.values.get("answer"),
        "status": final_state.values.get("status"),
        "clarification_question": final_state.values.get("clarification_question"),
        "summary": final_state.values.get("summary"),
        "user_profile": final_state.values.get("user_profile"),
    }
