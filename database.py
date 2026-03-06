import sqlite3
from datetime import datetime

def initialize_memory_db():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_input TEXT NOT NULL,
            bot_reply TEXT NOT NULL,
            source TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            speaker TEXT,
            message TEXT,
            timestamp TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

def initialize_user_profile():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            language TEXT DEFAULT 'tr',
            tone TEXT DEFAULT 'formal',
            interests TEXT DEFAULT '',
            sources TEXT DEFAULT 'json,wikipedia,chatgpt',
            email TEXT DEFAULT '',
            first_name TEXT DEFAULT '',
            last_name TEXT DEFAULT '',
            age INTEGER DEFAULT 0,
            birthdate TEXT DEFAULT '',
            chat_count INTEGER DEFAULT 0,
            password TEXT DEFAULT ''
        )
    """)
    # Fallback alters for backward compatibility
    columns = [
        ("email", "TEXT DEFAULT ''"), ("first_name", "TEXT DEFAULT ''"),
        ("last_name", "TEXT DEFAULT ''"), ("age", "INTEGER DEFAULT 0"),
        ("chat_count", "INTEGER DEFAULT 0"), ("birthdate", "TEXT DEFAULT ''"),
        ("password", "TEXT DEFAULT ''"), ("theme", "TEXT DEFAULT 'system'")
    ]
    for col_name, col_type in columns:
        try:
            cursor.execute(f"ALTER TABLE user_profile ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass 
            
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_user_profile_email ON user_profile(email)")
    conn.commit()
    conn.close()

def log_to_memory(user_input, bot_reply, source):
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO memory_log (timestamp, user_input, bot_reply, source) VALUES (?, ?, ?, ?)",
                   (timestamp, user_input, bot_reply, source))
    conn.commit()
    conn.close()