"""Persistent approved plans and completed sections for local research runs."""

from contextlib import closing, contextmanager
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3

from gpt_researcher.config.variables.base import BaseConfig


class SectionCheckpointError(RuntimeError):
    """The section checkpoint could not be safely used."""


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _task_fingerprint(task, tone, headers):
    config_keys = {
        *BaseConfig.__annotations__,
        "OPENAI_BASE_URL",
        "OLLAMA_BASE_URL",
        "RETRIEVER_ENDPOINT",
        "FAST_LLM_MODEL",
        "SMART_LLM_MODEL",
        "STRATEGIC_LLM_MODEL",
        "LLM_PROVIDER",
        "OPENAI_EMBEDDING_MODEL",
        "OLLAMA_EMBEDDING_MODEL",
        "EMBEDDING_PROVIDER",
    }
    config = {
        key: value
        for key, value in os.environ.items()
        if key in config_keys or key.startswith(("RETRIEVER_ARG_", "PUBMED_ARG_"))
    }
    config_path = os.environ.get("CONFIG_PATH")
    if config_path and Path(config_path).is_file():
        config["CONFIG_FILE_HASH"] = sha256(Path(config_path).read_bytes()).hexdigest()
    # Store only the hash, since the inputs can contain credentials.
    tone = tone.value if isinstance(tone, Enum) else tone
    try:
        identity = _encode([task, tone, headers, config])
    except (TypeError, ValueError) as error:
        raise SectionCheckpointError(
            "Checkpoint task settings must be JSON serializable"
        ) from error
    return sha256(identity.encode("utf-8")).hexdigest()


def has_section_draft(draft):
    return isinstance(draft, dict) and any(
        isinstance(text, str) and text.strip() for text in draft.values()
    )


class SectionCheckpoint:
    """Keep each section result once, keyed by its position in the approved plan."""

    def __init__(self, path, task, tone, headers):
        self.path = Path(path).expanduser().absolute()
        self.fingerprint = _task_fingerprint(task, tone, headers)
        with self._transaction() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            tables = db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            if version == 0 and not tables:
                db.execute(
                    "CREATE TABLE research_plan (id INTEGER PRIMARY KEY CHECK (id = 1), "
                    "fingerprint TEXT NOT NULL, plan TEXT)"
                )
                db.execute(
                    "CREATE TABLE section_drafts (position INTEGER PRIMARY KEY, draft TEXT)"
                )
                db.execute(
                    "INSERT INTO research_plan VALUES (1, ?, NULL)", (self.fingerprint,)
                )
                db.execute("PRAGMA user_version = 1")
            elif version != 1 or set(tables) != {
                ("research_plan",),
                ("section_drafts",),
            }:
                raise SectionCheckpointError("Unsupported section checkpoint format")
            stored = db.execute(
                "SELECT fingerprint FROM research_plan WHERE id = 1"
            ).fetchone()
            if stored != (self.fingerprint,):
                raise SectionCheckpointError(
                    "Checkpoint does not match this task or configuration"
                )

    @contextmanager
    def _transaction(self):
        try:
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                yield db
        except sqlite3.Error as error:
            raise SectionCheckpointError(
                f"Cannot use section checkpoint {self.path}: {error}"
            ) from error

    def _decode(self, raw):
        try:
            return json.loads(raw)
        except (TypeError, ValueError) as error:
            raise SectionCheckpointError("Malformed section checkpoint data") from error

    def load_plan(self):
        with self._transaction() as db:
            row = db.execute("SELECT plan FROM research_plan WHERE id = 1").fetchone()
        if row is None:
            raise SectionCheckpointError("Missing research plan row")
        if row[0] is None:
            return None
        plan = self._decode(row[0])
        if not isinstance(plan, dict) or not _valid_sections(plan.get("sections")):
            raise SectionCheckpointError("Invalid saved section plan")
        return {key: plan.get(key) for key in ("title", "date", "sections")}

    def save_plan(self, state):
        plan = {key: state.get(key) for key in ("title", "date", "sections")}
        if not _valid_sections(plan["sections"]):
            raise SectionCheckpointError("Expected a list of approved sections")
        serialized = _encode(plan)
        with self._transaction() as db:
            db.execute(
                "UPDATE research_plan SET plan = ? WHERE id = 1 AND plan IS NULL",
                (serialized,),
            )
            row = db.execute("SELECT plan FROM research_plan WHERE id = 1").fetchone()
            if row != (serialized,):
                raise SectionCheckpointError(
                    "Approved plan differs from the saved plan"
                )

    def completed_drafts(self, section_count):
        with self._transaction() as db:
            rows = db.execute(
                "SELECT position, draft FROM section_drafts WHERE draft IS NOT NULL"
            ).fetchall()
        drafts = {position: self._decode(raw) for position, raw in rows}
        if any(
            not isinstance(position, int)
            or not 0 <= position < section_count
            or not has_section_draft(draft)
            for position, draft in drafts.items()
        ):
            raise SectionCheckpointError("Invalid saved section draft")
        return drafts

    def record_result(self, position, draft):
        value = (
            json.dumps(draft, ensure_ascii=False) if has_section_draft(draft) else None
        )
        with self._transaction() as db:
            db.execute(
                "INSERT INTO section_drafts VALUES (?, ?) "
                "ON CONFLICT(position) DO UPDATE SET draft = excluded.draft "
                "WHERE section_drafts.draft IS NULL",
                (position, value),
            )
            saved = db.execute(
                "SELECT draft FROM section_drafts WHERE position = ?", (position,)
            ).fetchone()[0]
        return None if saved is None else self._decode(saved)


def _valid_sections(sections):
    return isinstance(sections, list) and all(
        isinstance(section, str) and section.strip() for section in sections
    )
