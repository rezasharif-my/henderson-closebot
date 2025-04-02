
from dataclasses import dataclass
from typing_extensions import TypedDict
from typing import Annotated, Optional, Any, Literal
from datetime import datetime

@dataclass(kw_only=True)
class UserData:
    """Stores user profile and lead information."""
    thread_id: str
    """LangGraph thread/session ID (unique identifier)."""

    name: Optional[str] = None
    """User's name (if detected or captured)."""

    platform: Optional[str] = None
    """Messaging platform: 'instagram' or 'messenger'."""

    email: Optional[str] = None
    """User's email address, if captured."""

    phone: Optional[str] = None
    """User's phone number, if captured."""

    last_active: Optional[datetime] = None
    """Timestamp of the user's last message or update."""
    
class Configurable(TypedDict, total=False):
    """Stores configurable settings for a chatbot session."""
    thread_id: Optional[str] = None
    """Unique thread ID for the conversation."""
    platform: Optional[str] = None


class Config(TypedDict, total=False):
    """Stores chatbot configuration settings."""
    configurable: Configurable
    """Nested configurable settings for the chatbot session."""