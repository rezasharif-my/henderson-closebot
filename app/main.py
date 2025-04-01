from fastapi import FastAPI, Request
from app.graph import run_langgraph_flow
from app.utils import extract_user_message, send_meta_reply
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

@app.get("/webhook")
def verify(mode: str = '', challenge: str = '', verify_token: str = ''):
    if verify_token == os.getenv("VERIFY_TOKEN"):
        return int(challenge)
    return {"error": "Verification failed"}

@app.post("/webhook")
async def webhook_listener(request: Request):
    data = await request.json()
    user_msg, sender_id, platform = extract_user_message(data)

    if user_msg:
        reply = await run_langgraph_flow(user_msg, sender_id)
        await send_meta_reply(sender_id, reply, platform)

    return {"status": "ok"}
