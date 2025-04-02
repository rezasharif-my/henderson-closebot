# 🤖 CloseBot – Jack Henderson’s AI Assistant

**CloseBot** is a custom AI chatbot built to emulate Jack Henderson’s confident, motivational, and direct communication style.  
It engages users across web, Instagram, and Messenger, answers property-related questions, captures leads, and maintains context using LangGraph’s memory system.

Built for **Henderson Advocacy**, this chatbot is your always-on, tone-consistent digital Jack — motivating leads and converting interest into action.

---

## 🧠 Powered By

- **LangGraph + LangChain** – Graph-based conversational logic
- **OpenAI GPT-4o** – Contextual tone-adaptive LLM
- **FastAPI** – Web & API backend
- **SQLite** – Local persistent chat/user store
- **Structured LLM Calls** – Using `pydantic` validation
- **Multi-Agent Nodes** – Modular reasoning and action steps

---

## 🖼️ Preview

> Demo screenshots or video (replace with your own)

![Chat Demo](docs/chat-demo.gif)
![Interface Preview](docs/ui-preview.png)

---

## 🔧 Project Structure

closebot/
├── app/
│   ├── main.py               # FastAPI app & routing
│   ├── graph.py              # LangGraph setup
│   ├── handlers.py           # Node functions (LLMs, logic, summaries)
│   ├── core.py               # core functions (generate answers ...)
│   ├── state.py              # State class for LangGraph
│   ├── db.py                 # SQLite setup and queries
│   └── utils.py              # Helper tools, environment setup
├── templates/chat.html       # Web UI
├── static/                   # JS / CSS if needed
├── requirements.txt
├── setup.py
├── README.md
└── docs/                     # Images, videos for documentation


---

## 🚀 Features

- ✅ Jack Henderson–style responses trained via structured LLM prompt engineering
- 💬 Multi-turn chat with built-in **LangGraph memory**
- 📥 Lead capture: phone & email with validation
- 🧠 Intent classification + clarification flow with **interrupt/resume**
- 🗃️ Full chat history & user data saved to SQLite
- ⚡ Motivator node: detects hesitation and delivers Jack’s signature “no-excuses” energy
- 🔁 Multiple messaging platforms (Messenger, Instagram, Web UI)

---

## 🔁 Node-by-Node Breakdown

| Node                         | Description                                                                     |
|------------------------------|---------------------------------------------------------------------------------|
| **`memory_loader_node`**     | Loads user info and summary from DB; resets conflicting fields                  |
| **`intent_classifier_node`** | Classifies user message (faq, invest, smalltalk, etc.)                          |
| **`faq_matcher_node`**       | Matches common questions from a dictionary                                      |
| **`clarification_node`**     | If unclear, generates a follow-up question and pauses graph                     |
| **`jack_reply_generator_node`** | Jack-styled response, using previous summary for tone & personalization      |
| **`lead_capture_node`**      | Extracts email/phone and validates before saving                                |
| **`motivator_node`**         | Adds motivating quote if user's intent seems hesitant                           |
| **`summary_node`**           | Generates & stores one-paragraph chat summary                                   |
| **`user_save_node`**         | Final step to persist profile updates                                           |

---

## 💻 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/rezasharif-my/henderson-closebot.git && cd closebot

# 2. Create a virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
uvicorn app.main:app --reload

Open your browser at http://localhost:8000 to test the chatbot UI.

🎯 API Endpoint

POST /message
Content-Type: application/json

{
  "sender": {
    "id": "1234567890123456"
  },
  "platform": "messenger",
  "message": {
    "text": "Hey Jack, I’m interested in investing!"
  }
}


Optional Web Testing Flow:
	•	Open browser and use the chat UI to simulate interaction
	•	Session is persisted via cookies (reset with the red 🔄 Reset Chat button)


## 📂 Database Schema

SQLite tables include:
	•	users – Stores thread_id, name, email, phone, platform
	•	chat_summaries – One-paragraph summaries per thread
	•	chat_history – Archived messages and timestamps


## 📄 License

MIT – Free to use, modify, and extend.