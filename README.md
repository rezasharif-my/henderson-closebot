# Henderson CloseBot

This is a LangGraph-powered chatbot built for Henderson to simulate Jack Henderson's persona across Facebook and Instagram.

## Features
- Multi-channel message handling (Meta API)
- Jack’s motivational, no-nonsense tone
- LangGraph flow with memory and intent routing
- Lead capture (email & phone) with validation
- SQLite storage for qualified leads

## How to Run
1. Clone this repo
2. Setup `.env` with your OpenAI & Meta API keys
3. Run `uvicorn app.main:app --reload`
4. Use ngrok to expose `/webhook` endpoint
5. Test by messaging the linked Instagram or Messenger page

## Tech Stack
- Python, FastAPI
- LangGraph, OpenAI GPT-4o
- SQLite
- Meta Graph API (Instagram + Facebook)

## License
MIT