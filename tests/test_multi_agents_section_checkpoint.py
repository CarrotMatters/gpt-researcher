"""Exercise section recovery through the real workflow in separate processes."""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest import TestCase
from unittest.mock import patch

from gpt_researcher.utils.enum import Tone
from multi_agents.agents.orchestrator import ChiefEditorAgent
from multi_agents.agents.researcher import ResearchAgent
from multi_agents.memory.section_checkpoint import (
    SectionCheckpoint,
    SectionCheckpointError,
)


TASK = {"query": "Demo research", "publish_formats": {"markdown": True}}
SECTIONS = ["Evidence", "Limitations", "Takeaways"]


async def run_sample(directory, mode):
    directory = Path(directory)
    calls = []

    class DemoEditor(ChiefEditorAgent):
        def _create_output_directory(self):
            path = directory / mode
            path.mkdir()
            return str(path)

        def _initialize_agents(self):
            agents = super()._initialize_agents()

            def browser(_):
                calls.append("browser")
                return {"initial_research": "Sample"}

            def planner(_):
                calls.append("planner")
                return {
                    "sections": SECTIONS,
                    "title": "Sample report",
                    "date": "2026-11-17",
                }

            agents["research"].run_initial_research = browser
            agents["editor"].plan_research = planner
            agents["human"].review_plan = lambda _: {"human_feedback": None}
            agents["writer"].run = lambda _: {
                "headers": {
                    "title": "Sample report",
                    "date": "Date",
                    "introduction": "Introduction",
                    "table_of_contents": "Contents",
                    "conclusion": "Conclusion",
                    "references": "References",
                },
                "table_of_contents": "\n".join(f"- {name}" for name in SECTIONS),
                "introduction": "An offline workflow exercise.",
                "conclusion": "This report contains the available section drafts.",
            }
            agents["fact_checker"].run = lambda _: {"fact_check_notes": None}
            agents["visualizer"].run = lambda _: {"diagrams": []}
            return agents

    async def research(_self, *, query, **_kwargs):
        calls.append(query)
        await asyncio.sleep(0.05 if query == "Limitations" else 0.35)
        if query == "Limitations":
            if mode == "initial":
                raise TimeoutError("Injected timeout")
            if mode == "cancel":
                await asyncio.sleep(0.5)
                raise asyncio.CancelledError
        return f"## {query}\nDraft for {query}."

    checkpoint = None if mode == "full" else directory / "research.sqlite"
    started = time.perf_counter()
    with patch.object(ResearchAgent, "research", research):
        try:
            result = await DemoEditor(TASK, tone=Tone.Objective).run_research_task(
                checkpoint_path=checkpoint
            )
        except (Exception, asyncio.CancelledError) as error:
            if mode != "cancel" or not isinstance(
                error.__cause__ or error, asyncio.CancelledError
            ):
                raise
            result = None
    elapsed = time.perf_counter() - started
    if result is None:
        assert not list((directory / mode).glob("*.md"))
        summary = {"cancelled": True, "calls": calls, "seconds": elapsed}
    else:
        (report,) = (directory / mode).glob("*.md")
        summary = {
            "calls": calls,
            "seconds": elapsed,
            "sections": [next(iter(draft)) for draft in result["research_data"]],
            "failed_sections": result["failed_sections"],
            "report": report.read_text(encoding="utf-8"),
        }
    (directory / f"{mode}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def sample_in_process(directory, mode):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_multi_agents_section_checkpoint",
            str(directory),
            mode,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode:
        raise AssertionError(result.stdout + "\n" + result.stderr)
    return json.loads((Path(directory) / f"{mode}.json").read_text(encoding="utf-8"))


class SectionRecoveryTests(TestCase):
    def test_process_restart_reuses_success_and_preserves_section_order(self):
        with TemporaryDirectory() as directory:
            first = sample_in_process(directory, "initial")
            full = sample_in_process(directory, "full")
            resumed = sample_in_process(directory, "resume")
            repeated = sample_in_process(directory, "repeat")

        self.assertEqual(first["failed_sections"], ["Limitations"])
        self.assertEqual(first["sections"], ["Evidence", "Takeaways"])
        self.assertCountEqual(full["calls"], ["browser", "planner", *SECTIONS])
        self.assertEqual(resumed["calls"], ["Limitations"])
        self.assertEqual(repeated["calls"], [])
        self.assertEqual(resumed["sections"], SECTIONS)
        self.assertEqual(resumed["report"], full["report"])
        self.assertEqual(repeated["report"], resumed["report"])
        self.assertEqual(resumed["failed_sections"], [])

    def test_completed_siblings_are_reused_after_cancellation(self):
        with TemporaryDirectory() as directory:
            cancelled = sample_in_process(directory, "cancel")
            resumed = sample_in_process(directory, "resume")

        self.assertTrue(cancelled["cancelled"])
        self.assertEqual(resumed["calls"], ["Limitations"])
        self.assertEqual(resumed["sections"], SECTIONS)

    def test_task_and_configuration_mismatch_do_not_reuse_a_plan(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "research.sqlite"
            original = SectionCheckpoint(path, TASK, None, {})
            original.save_plan({"sections": SECTIONS})
            with self.assertRaisesRegex(SectionCheckpointError, "does not match"):
                SectionCheckpoint(path, {**TASK, "query": "Other task"}, None, {})
            with patch.dict("os.environ", {"RETRIEVER": "other-retriever"}):
                with self.assertRaisesRegex(SectionCheckpointError, "does not match"):
                    SectionCheckpoint(path, TASK, None, {})
            self.assertEqual(original.load_plan()["sections"], SECTIONS)
            other = SectionCheckpoint(Path(directory) / "other.sqlite", TASK, None, {})
            self.assertIsNone(other.load_plan())

    def test_checkpoint_errors_are_explicit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "corrupt.sqlite"
            corrupt.write_bytes(b"not a database")
            for path in (corrupt, root / "missing" / "checkpoint.sqlite"):
                with self.subTest(path=path), self.assertRaises(SectionCheckpointError):
                    SectionCheckpoint(path, TASK, None, {})

    def test_config_file_content_is_part_of_checkpoint_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text('{"TOTAL_WORDS": 300}', encoding="utf-8")
            with patch.dict("os.environ", {"CONFIG_PATH": str(config)}):
                SectionCheckpoint(root / "research.sqlite", TASK, None, {})
                config.write_text('{"TOTAL_WORDS": 600}', encoding="utf-8")
                with self.assertRaisesRegex(SectionCheckpointError, "does not match"):
                    SectionCheckpoint(root / "research.sqlite", TASK, None, {})

    def test_existing_success_wins_over_later_failure_or_duplicate(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite"
            first = SectionCheckpoint(path, TASK, None, {})
            second = SectionCheckpoint(path, TASK, None, {})
            first.save_plan({"sections": SECTIONS})
            self.assertIsNone(first.record_result(0, None))
            saved = {"Evidence": "First successful draft"}
            self.assertEqual(second.record_result(0, saved), saved)
            self.assertEqual(first.record_result(0, {"Evidence": "Later draft"}), saved)
            self.assertEqual(second.record_result(0, None), saved)
            self.assertEqual(first.completed_drafts(len(SECTIONS)), {0: saved})


if __name__ == "__main__":
    asyncio.run(run_sample(sys.argv[1], sys.argv[2]))
