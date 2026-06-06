"""
会话存储服务
使用 JSON 文件存储用户和会话数据
"""
import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# 数据文件路径
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sessions.json")


class StorageService:
    """会话存储服务类"""

    def __init__(self):
        self._data = None
        self._ensure_data_file()

    def _ensure_data_file(self):
        """确保数据文件存在"""
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        if not os.path.exists(DATA_FILE):
            self._data = {"users": {}}
            self._save_data()
        else:
            self._load_data()

    def _load_data(self):
        """加载数据文件"""
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        except Exception as e:
            logger.error(f"加载会话数据失败: {e}")
            self._data = {"users": {}}

    def _save_data(self):
        """保存数据文件"""
        try:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存会话数据失败: {e}")

    def get_or_create_user(self, user_id: str) -> Dict:
        """获取或创建用户"""
        if user_id not in self._data["users"]:
            self._data["users"][user_id] = {
                "sessions": {},
                "created_at": datetime.now().isoformat()
            }
            self._save_data()
            logger.info(f"创建新用户: {user_id}")
        return self._data["users"][user_id]

    def create_session(self, user_id: str, title: str = "新会话") -> str:
        """创建新会话"""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        user = self.get_or_create_user(user_id)

        user["sessions"][session_id] = {
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
        self._save_data()
        logger.info(f"用户 {user_id} 创建新会话: {session_id}")
        return session_id

    def get_session(self, user_id: str, session_id: str) -> Optional[Dict]:
        """获取指定会话"""
        user = self._data["users"].get(user_id)
        if not user:
            return None
        return user["sessions"].get(session_id)

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """获取用户所有会话列表"""
        user = self._data["users"].get(user_id)
        if not user:
            return []

        sessions = []
        for session_id, session_data in user["sessions"].items():
            sessions.append({
                "session_id": session_id,
                "title": session_data["title"],
                "updated_at": session_data["updated_at"],
                "message_count": len(session_data["messages"]),
                "last_message": session_data["messages"][-1] if session_data["messages"] else None
            })

        # 按更新时间排序，最新的在前
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions

    def save_message(self, user_id: str, session_id: str, role: str, content: str):
        """保存消息到会话"""
        session = self.get_session(user_id, session_id)
        if not session:
            logger.warning(f"会话不存在: {user_id}/{session_id}")
            return False

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        session["messages"].append(message)
        session["updated_at"] = datetime.now().isoformat()

        # 如果是第一条用户消息，更新会话标题
        if role == "user" and len(session["messages"]) == 1:
            session["title"] = content[:20] + "..." if len(content) > 20 else content

        self._save_data()
        return True

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除会话"""
        user = self._data["users"].get(user_id)
        if not user:
            return False

        if session_id not in user["sessions"]:
            return False

        del user["sessions"][session_id]
        self._save_data()
        logger.info(f"用户 {user_id} 删除会话: {session_id}")
        return True

    def rename_session(self, user_id: str, session_id: str, new_title: str) -> bool:
        """重命名会话"""
        session = self.get_session(user_id, session_id)
        if not session:
            return False

        session["title"] = new_title
        session["updated_at"] = datetime.now().isoformat()
        self._save_data()
        return True

    def get_session_messages(self, user_id: str, session_id: str) -> List[Dict]:
        """获取会话的所有消息"""
        session = self.get_session(user_id, session_id)
        if not session:
            return []
        return session["messages"]


# 模块级单例
_storage_service = None


def get_storage_service() -> StorageService:
    """获取存储服务单例"""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service