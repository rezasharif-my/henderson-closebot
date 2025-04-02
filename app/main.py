from fastapi import FastAPI, Request ,Form
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import uuid
import os

from core import generate_answer

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# Enable sessions
app.add_middleware(SessionMiddleware, secret_key="super-secret-reza-key")

# =========================================
# Pydantic model ( API input format)
# =========================================
class MessengerInput(BaseModel):
    sender: dict
    recipient: dict
    message: dict
    platform: str = "web" # e.g. "instagram", "facebook", "web"

# =========================================
# UI Route – GET (Initialize session)
# =========================================
@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    if "thread_id" not in request.session:
        request.session["thread_id"] = f"web_{str(uuid.uuid4())[:8]}"
    chat_history = request.session.get("chat_history", [])
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "chat": chat_history
    })

# =========================================
# UI Route – POST (Handle chat input)
# =========================================
@app.post("/", response_class=HTMLResponse)
async def chat_post(request: Request, message: str = Form(...)):
    thread_id = request.session.get("thread_id")
    result = generate_answer(message, thread_id=thread_id, platform="web")
    
    # Initialize chat history in session if not exist
    if "chat_history" not in request.session:
        request.session["chat_history"] = []
    request.session["chat_history"].append({"from": "user", "msg": message})
    request.session["chat_history"].append({"from": "jack", "msg": result["answer"]})


    return templates.TemplateResponse("chat.html", {
        "request": request,
        "chat": request.session["chat_history"]
    })
    
# =========================================
# Reset session – clear all session data
# =========================================
@app.post("/reset")
async def reset_session(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)

# =========================================
# API Endpoint – for programmatic testing
# =========================================
# @app.post("/chat")
# async def handle_chat_api(payload: MessengerInput):
#     thread_id = f"{payload.platform}_{payload.sender['id']}"

#     initial_state = State(
#         messages=[{"type": "human", "content": payload.message["text"]}],
#         thread_id=thread_id,
#         user_profile={"platform": payload.platform}
#     )

#     try:
#         result = graph.invoke(initial_state)
#         return {
#             "reply": result.answer,
#             "status": result.status,
#             "summary": result.summary,
#             "user_profile": result.user_profile
#         }
#     except Exception as e:
#         return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)