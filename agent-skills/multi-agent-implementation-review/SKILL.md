---
name: multi-agent-implementation-review
description: Coordinate a multi-agent coding workflow with one orchestrator, bounded implementation workers, and read-only review or verification agents. Use when the user explicitly asks for multiple agents, parallel execution, an implementation agent plus review agents, or an agent-first implementation and review loop.
---

# Multi-Agent Implementation Review

Use this skill only when the user explicitly asks for sub-agents, delegation, parallel work, reviewer agents, or a harness-style implementation and review loop.

The orchestrator should stay on the critical path. Delegate only work that is concrete, bounded, and can run in parallel without blocking the next local step.

## Default Topology

- Keep one orchestrator responsible for the final plan, integration, and verification.
- Use writer agents for bounded implementation slices with disjoint ownership.
- Use read-only review agents for regressions, missing tests, architecture checks, or acceptance-criteria validation.
- Start small. The default is one implementation worker plus one review agent.
- Add more agents only when the write scopes are clearly separate.

## Workflow

1. Inspect the codebase and the task locally before delegating.
2. Decide which immediate blocking step stays with the orchestrator.
3. Split only independent sidecar work that can run in parallel.
4. Give each sub-agent a tight brief with:
   - the goal and acceptance criteria
   - explicit file or module ownership
   - verification expectations
   - an instruction not to revert others' edits
   - a request to report changed files or findings clearly
5. Keep doing non-overlapping local work while sub-agents run.
6. Wait for a sub-agent only when the next critical step depends on its result.
7. Review returned changes or findings, integrate carefully, and resolve conflicts.
8. Run verification locally: tests, lint, typecheck, or UI checks appropriate to the task.
9. If the change is risky, run one more read-only review pass before finishing.

## Decomposition Rules

- Split by ownership, not by vague notions of "help."
- Never assign the same hot files to multiple writer agents.
- Keep architecture-sensitive refactors under a single owner when possible.
- Prefer review agents for:
  - regression scanning
  - missing-test analysis
  - architecture or boundary checks
  - docs and acceptance-criteria checks
- If the task is small, localized, or blocked on one result, do not spawn extra agents.

## Recommended Patterns

- Small feature or bugfix:
  - one implementation worker
  - one read-only reviewer
- Cross-stack feature:
  - one backend worker
  - one frontend worker
  - one read-only reviewer
- Quality-focused change:
  - one implementation worker
  - one test or eval worker
  - one read-only reviewer
- Review-only request:
  - one or two read-only reviewers with distinct focus areas

## Sub-Agent Prompt Requirements

Every writer prompt should include:

- the user goal in one or two lines
- the owned files, directories, or module boundaries
- success criteria and tests to run
- "You are not alone in the codebase; do not revert others' edits."
- a request to list changed files in the final handoff

Every reviewer prompt should include:

- a read-only constraint unless edits are explicitly requested
- the review focus, such as regressions, test gaps, architecture, or spec alignment
- the expected output format, for example findings ordered by severity
- a request to cite specific files and lines when possible

For reusable prompt scaffolds, read [references/prompt-templates.md](references/prompt-templates.md).

## Verification and Handoff

- Verify the integrated result, not just the delegated fragments.
- Prefer narrow, task-relevant checks first, then broader checks if shared infrastructure changed.
- Summarize what changed, what was validated, and any residual risks.
- If the platform does not support sub-agents, follow the same split mentally and execute the workflow sequentially.
