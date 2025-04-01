import sqlite3
from datetime import datetime

DB_PATH = "app/database/user_info.db"

# Initialize DB + tables
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # User profile & lead info
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT UNIQUE,
        name TEXT,
        platform TEXT,
        email TEXT,
        phone TEXT,
        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Chat summary per session
    c.execute('''CREATE TABLE IF NOT EXISTS chat_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT,
        summary TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Full chat logs
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT,
        role TEXT,  -- 'user' or 'assistant'
        message TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()
    conn.close()

# Save message to history
def save_message(thread_id: str, role: str, message: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO chat_history (thread_id, role, message)
        VALUES (?, ?, ?)
    """, (thread_id, role, message))
    conn.commit()
    conn.close()

# Save or update user
def save_user(thread_id, name=None, platform=None, email=None, phone=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO users (thread_id, name, platform, email, phone)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            name=COALESCE(?, name),
            platform=COALESCE(?, platform),
            email=COALESCE(?, email),
            phone=COALESCE(?, phone),
            last_active=CURRENT_TIMESTAMP
    """, (thread_id, name, platform, email, phone, name, platform, email, phone))
    conn.commit()
    conn.close()

# Fetch last messages
def get_recent_chat(thread_id, limit=5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT role, message FROM chat_history
        WHERE thread_id=?
        ORDER BY timestamp DESC LIMIT ?
    """, (thread_id, limit))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]  # reverse to chronological

# Save chat summary
def save_chat_summary(thread_id, summary):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO chat_summaries (thread_id, summary)
        VALUES (?, ?)
    """, (thread_id, summary))
    conn.commit()
    conn.close()
# Get user by thread_id
def get_user_by_thread_id(thread_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT name, platform, email, phone FROM users
        WHERE thread_id = ?
    """, (thread_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "name": row[0],
            "platform": row[1],
            "email": row[2],
            "phone": row[3],
        }
    return None

# Get latest summary by thread_id
def get_latest_summary(thread_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT summary FROM chat_summaries
        WHERE thread_id = ?
        ORDER BY timestamp DESC LIMIT 1
    """, (thread_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# Run this once to create tables
# if __name__ == "__main__":
#     init_db()
#     print("Database initialized ")