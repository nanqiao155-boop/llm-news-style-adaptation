from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
DATA_DIR = DEMO_DIR / "demo_data"
ASSETS_DIR = DEMO_DIR / "assets"
LOCAL_ENV_PATH = DEMO_DIR / ".env.local"

SYSTEM_TITLE = "LLM News Writing–Review–Revision Demo"
REPLAY_NOTICE = "Synthetic fixtures only; excluded from every reported experiment."

STEP_NAMES = ("素材输入", "Writer", "Reviewer", "Reviser", "质量评测")
STEP_STATES = {"pending", "active", "completed", "skipped", "failed"}

EDITORIAL_DIMENSIONS = {
    "factual_grounding": 30,
    "key_information_retention": 15,
    "title_quality": 15,
    "formality_objectivity": 15,
    "structure_formatting": 10,
    "conciseness_naturalness": 15,
}


def load_local_env(path: Path = LOCAL_ENV_PATH) -> bool:
    """Load local private settings without overriding shell environment variables."""
    if not path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        raise RuntimeError("缺少 python-dotenv，请安装 demo/requirements.txt。") from None
    return bool(load_dotenv(dotenv_path=path, override=False, encoding="utf-8", interpolate=False))


@dataclass(frozen=True)
class APISettings:
    api_base: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float = 90.0

    @classmethod
    def from_env(cls, env_file: Path | None = LOCAL_ENV_PATH) -> "APISettings":
        if env_file is not None:
            load_local_env(env_file)
        return cls(
            api_base=os.environ.get("LLM_JUDGE_API_BASE", "").strip(),
            api_key=os.environ.get("LLM_JUDGE_API_KEY", "").strip(),
            model=os.environ.get("LLM_JUDGE_MODEL", "").strip(),
        )

    @property
    def missing_variables(self) -> tuple[str, ...]:
        values = {
            "LLM_JUDGE_API_BASE": self.api_base,
            "LLM_JUDGE_API_KEY": self.api_key,
            "LLM_JUDGE_MODEL": self.model,
        }
        return tuple(name for name, value in values.items() if not value)

    @property
    def is_configured(self) -> bool:
        return not self.missing_variables


@dataclass(frozen=True)
class WriterAPISettings:
    api_base: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float = 90.0
    api_base_from_judge: bool = False
    api_key_from_judge: bool = field(default=False, repr=False)

    @classmethod
    def from_env(
        cls,
        judge: APISettings | None = None,
        env_file: Path | None = LOCAL_ENV_PATH,
    ) -> "WriterAPISettings":
        if env_file is not None:
            load_local_env(env_file)
        judge = judge or APISettings.from_env(env_file=None)
        writer_base = os.environ.get("LLM_WRITER_API_BASE", "").strip()
        writer_key = os.environ.get("LLM_WRITER_API_KEY", "").strip()
        return cls(
            api_base=writer_base or judge.api_base,
            api_key=writer_key or judge.api_key,
            model=os.environ.get("LLM_WRITER_MODEL", "").strip(),
            api_base_from_judge=not bool(writer_base),
            api_key_from_judge=not bool(writer_key),
        )

    @property
    def missing_variables(self) -> tuple[str, ...]:
        missing = []
        if not self.api_base:
            missing.append("LLM_WRITER_API_BASE or LLM_JUDGE_API_BASE")
        if not self.api_key:
            missing.append("LLM_WRITER_API_KEY or LLM_JUDGE_API_KEY")
        if not self.model:
            missing.append("LLM_WRITER_MODEL")
        return tuple(missing)

    @property
    def is_configured(self) -> bool:
        return not self.missing_variables
