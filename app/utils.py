import os, getpass
import uuid
from dotenv import load_dotenv
load_dotenv()

def generate_uuid():
    """Generate a unique identifier."""
    return str(uuid.uuid4())

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

def get_environment_ready():
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
def reset_state_fields(state: State, fields_to_reset: List[str]) -> State:
    """Resets only specific fields of the chatbot state while keeping the rest."""
    for field in fields_to_reset:
        if hasattr(state, field):  # Check if the attribute exists
            setattr(state, field, None if field != "messages" else [])
    return state