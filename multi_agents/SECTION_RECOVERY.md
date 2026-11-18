# Resume completed research sections

The multi-agent workflow keeps successful section drafts when sibling tasks fail. Published reports list failed sections. If every section fails, the workflow raises an error before publication.

To retain drafts across process restarts, pass a local SQLite checkpoint:

```python
from multi_agents.agents import ChiefEditorAgent

chief = ChiefEditorAgent(task)
result = await chief.run_research_task(checkpoint_path="research.sqlite")
```

From the repository root, the CLI uses the current `multi_agents/task.json`:

```powershell
python -m multi_agents.main --checkpoint research.sqlite
```

Run the same command again with the same task and configuration to resume. The workflow loads its approved outline, reuses successful drafts, and researches unfinished sections. The writer, fact checker, visualizer and publisher run again. Results retain the approved section order. Use a separate checkpoint for each research task.

The checkpoint stores the outline and successful section drafts using SQLite transactions. Relevant task and environment settings are bound to the checkpoint with a fingerprint. Mismatched settings, an unsupported checkpoint format, and database failures raise `SectionCheckpointError`. Configuration-file content is included in the fingerprint.

The checkpoint contains report drafts in plain text. When two processes overlap on one checkpoint, the first successful draft committed for a position wins. Concurrent processes may still make duplicate external calls. Sequential recovery reuses previously committed drafts.

## Verification

```powershell
$env:LITELLM_LOCAL_MODEL_COST_MAP = 'True'
python -m unittest tests.test_multi_agents_section_checkpoint tests.test_multi_agents_section_failures
```

These tests run real workflow graphs with fixed research and model functions. Separate processes verify that a three-section run with one failure recovers by executing only the failed section. Repeating recovery executes no research sections. Additional checks cover cancellation, task isolation, changed configuration files, database errors and repeated writes.
