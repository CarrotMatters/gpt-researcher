import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from multi_agents.agents.orchestrator import ChiefEditorAgent
from multi_agents.agents.researcher import ResearchAgent
from multi_agents.agents.reviewer import ReviewerAgent


class SectionFailureTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.output_dir = Path(self.enterContext(TemporaryDirectory()))

    async def run_report(self, *sections):
        task = {"query": "Offline report", "publish_formats": {"markdown": True}}
        with patch.object(
            ChiefEditorAgent, "_create_output_directory", return_value=str(self.output_dir)
        ):
            chief = ChiefEditorAgent(task)

        async def research(agent, *, query, **kwargs):
            if query == "Search timeout":
                raise TimeoutError("Simulated search timeout")
            return f"## {query}\nDraft for {query}."

        async def review(agent, state):
            if state["topic"] == "Review failure":
                raise RuntimeError("Simulated review failure")
            if state["topic"] == "Cancelled":
                raise asyncio.CancelledError
            return {"review": None}

        agents = chief._initialize_agents()
        agents["research"].run_initial_research = lambda state: {"initial_research": "Fixture"}
        agents["editor"].plan_research = lambda state: {"sections": list(sections)}
        agents["human"].review_plan = lambda state: {"human_feedback": None}
        agents["writer"].run = lambda state: {
            "headers": {
                "title": task["query"],
                "date": "Date",
                "introduction": "Introduction",
                "table_of_contents": "Contents",
                "conclusion": "Conclusion",
                "references": "References",
            },
            "date": "2026-11-17",
            "introduction": "Offline workflow demonstration using fixed sample sections.",
            "table_of_contents": "\n".join(f"- {section}" for section in state["sections"]),
            "conclusion": "This report records the outcome of the sample research tasks.",
        }
        agents["fact_checker"].run = lambda state: {"fact_check_notes": None}
        agents["visualizer"].run = lambda state: {"diagrams": []}
        graph = chief._create_workflow(agents).compile()

        with patch.object(ResearchAgent, "research", research), patch.object(
            ReviewerAgent, "run", review
        ):
            return await graph.ainvoke({"task": task})

    async def test_partial_report_preserves_successes_and_lists_failed_sections(self):
        result = await self.run_report("Evidence", "Review failure", "Limitations", "Search timeout")

        self.assertEqual(
            result["research_data"],
            [
                {"Evidence": "## Evidence\nDraft for Evidence."},
                {"Limitations": "## Limitations\nDraft for Limitations."},
            ],
        )
        self.assertEqual(result["failed_sections"], ["Review failure", "Search timeout"])
        (report_path,) = self.output_dir.glob("*.md")
        report = report_path.read_text(encoding="utf-8")
        self.assertEqual(report, result["report"])
        self.assertIn("## Incomplete report", report)
        self.assertIn("- Review failure\n- Search timeout", report)
        self.assertLess(report.index("## Evidence"), report.index("## Limitations"))
        self.assertNotIn("Simulated review failure", report)

    async def test_all_failed_sections_stop_publication(self):
        with self.assertRaisesRegex(RuntimeError, "All report sections failed"):
            await self.run_report("Review failure", "Search timeout")

        self.assertEqual(list(self.output_dir.iterdir()), [])

    async def test_successful_report_has_no_failure_notice(self):
        result = await self.run_report("Evidence", "Limitations")

        self.assertNotIn("## Incomplete report", result["report"])

    async def test_cancellation_propagates(self):
        with self.assertRaises((Exception, asyncio.CancelledError)) as raised:
            await self.run_report("Evidence", "Cancelled")

        error = raised.exception
        self.assertIsInstance(error.__cause__ or error, asyncio.CancelledError)
        self.assertEqual(list(self.output_dir.iterdir()), [])
