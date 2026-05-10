"""
SQLite-backed chat history store.
Stores conversation turns with session management.
"""

import aiosqlite
import json
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from app.core.config import settings
from app.core.exceptions import SessionNotFoundError
from app.core.logging_config import get_logger
from app.schemas.chat import ChatMessage

logger = get_logger(__name__)

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_updated TEXT NOT NULL
)
"""

CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
)
"""

CREATE_IDX = "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"


class ChatHistoryDB:
    def __init__(self):
        self._db_path = settings.CHAT_HISTORY_DB_PATH

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(CREATE_SESSIONS_TABLE)
            await db.execute(CREATE_MESSAGES_TABLE)
            await db.execute(CREATE_IDX)
            await db.commit()
        logger.info("Chat history database initialized", extra={"db_path": self._db_path})

    async def close(self) -> None:
        logger.info("Chat history DB connection closed")

    async def create_session(self, session_id: str) -> None:
        now = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, last_updated) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
            await db.commit()
        logger.debug("Session created", extra={"session_id": session_id})

    async def add_messages(self, session_id: str, messages: List[ChatMessage]) -> None:
        now = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            for msg in messages:
                await db.execute(
                    "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (session_id, msg.role, msg.content, msg.timestamp.isoformat() if msg.timestamp else now),
                )
            await db.execute(
                "UPDATE sessions SET last_updated = ? WHERE session_id = ?", (now, session_id)
            )
            await db.commit()

    async def get_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        limit = limit or settings.MAX_HISTORY_TURNS * 2
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [ChatMessage(role=r[0], content=r[1], timestamp=datetime.fromisoformat(r[2])) for r in reversed(rows)]

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT created_at, last_updated FROM sessions WHERE session_id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            raise SessionNotFoundError(session_id)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ) as cursor:
                count_row = await cursor.fetchone()
        return {
            "created_at": datetime.fromisoformat(row[0]),
            "last_updated": datetime.fromisoformat(row[1]),
            "total_messages": count_row[0] if count_row else 0,
        }

    async def session_exists(self, session_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ) as cursor:
                return await cursor.fetchone() is not None

    async def cleanup_old_sessions(self) -> int:
        cutoff = (datetime.utcnow() - timedelta(days=settings.HISTORY_RETENTION_DAYS)).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT session_id FROM sessions WHERE last_updated < ?", (cutoff,)
            ) as cursor:
                old = [r[0] for r in await cursor.fetchall()]
            for sid in old:
                await db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                await db.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            await db.commit()
        logger.info("Old sessions cleaned up", extra={"count": len(old)})
        return len(old)
