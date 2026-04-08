from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    faq_dir: str = os.getenv("FAQ_DIR", "knowledge")
    faq_min_score: int = int(os.getenv("FAQ_MIN_SCORE", "1"))
    server_name: str = os.getenv("SERVER_NAME", "faq-agent")


settings = Settings()
