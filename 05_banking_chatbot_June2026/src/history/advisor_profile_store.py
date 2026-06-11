"""
AdvisorProfileStore — lưu/load collected_info của advisor per UserID + Domain.

Storage: data/conversations/{user_id}_profiles.json
Format:
{
  "user_id": "UID001",
  "profiles": {
    "loan": {
      "muc_dich_vay": "mua nhà",
      "so_tien_can_vay": "4 tỷ",
      ...
      "_updated_at": "2026-06-07T14:00:00"
    },
    "credit_card": { ... }
  }
}
"""
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from config.settings import settings


class AdvisorProfileStore:
    """Thread-safe store for saving/loading advisor collected_info per user+domain."""

    def __init__(self):
        self._lock = threading.Lock()

    def _profile_path(self, user_id: str) -> Path:
        return Path(settings.conversations_dir) / f"{user_id}_profiles.json"

    def _load_raw(self, user_id: str) -> dict:
        path = self._profile_path(user_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Cannot read profile {path}: {e}")
        return {"user_id": user_id, "profiles": {}}

    def _save_raw(self, data: dict):
        user_id = data["user_id"]
        path = self._profile_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_profile(self, user_id: str, domain: str, collected_info: dict):
        """
        Lưu collected_info cho user+domain.
        Gọi sau khi advisor recommendation hoàn thành.
        """
        with self._lock:
            data = self._load_raw(user_id)
            profile = {k: v for k, v in collected_info.items()}
            profile["_updated_at"] = datetime.now().isoformat()
            data["profiles"][domain] = profile
            self._save_raw(data)
        logger.info(
            f"Saved advisor profile: user={user_id}, domain={domain}, "
            f"fields={[k for k in collected_info]}"
        )

    def load_profile(self, user_id: str, domain: str) -> Optional[dict]:
        """
        Load collected_info cho user+domain.
        Trả về dict fields (không gồm _updated_at) hoặc None nếu chưa có.
        """
        data = self._load_raw(user_id)
        profile = data.get("profiles", {}).get(domain)
        if not profile:
            return None
        # Loại bỏ metadata field trước khi trả về
        clean = {k: v for k, v in profile.items() if not k.startswith("_")}
        return clean if clean else None

    def update_profile(self, user_id: str, domain: str, updates: dict):
        """
        Merge updates vào profile hiện có.
        """
        with self._lock:
            data = self._load_raw(user_id)
            existing = data.get("profiles", {}).get(domain, {})
            existing.update(updates)
            existing["_updated_at"] = datetime.now().isoformat()
            if "profiles" not in data:
                data["profiles"] = {}
            data["profiles"][domain] = existing
            self._save_raw(data)
        logger.info(
            f"Updated advisor profile: user={user_id}, domain={domain}, updates={list(updates.keys())}"
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
_profile_store: Optional[AdvisorProfileStore] = None


def get_profile_store() -> AdvisorProfileStore:
    global _profile_store
    if _profile_store is None:
        _profile_store = AdvisorProfileStore()
    return _profile_store
