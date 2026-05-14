# Prompt Templates

Use these as scaffolds, not scripts. Tighten ownership and acceptance criteria for the current task before sending them to sub-agents.

## Implementation Worker

```text
You are responsible for implementing a bounded slice of work.

Task:
<one or two lines>

Owned scope:
<files, directories, or module boundaries>

Acceptance criteria:
- <criterion>
- <criterion>

Verification:
- Run <tests/checks> if available
- Report any blockers immediately

Constraints:
- You are not alone in the codebase; do not revert others' edits.
- Stay within the owned scope unless a small supporting change is required.
- In your final handoff, list the files you changed and a short summary of why.
```

## Read-Only Reviewer

```text
Do a read-only review of the current changes.

Focus:
- <regressions | missing tests | architecture boundaries | spec alignment>

Review scope:
<diff, files, or modules>

Output:
- Findings ordered by severity
- Cite file paths and line numbers when possible
- Call out missing tests or risky assumptions

Do not edit files unless I explicitly ask for patches.
```

## Backend + Frontend Split

```text
Worker 1 owns:
- backend/API changes under <path>

Worker 2 owns:
- frontend/UI changes under <path>

Shared rule:
- Do not revert others' edits.
- If you discover a required cross-boundary change, report it instead of expanding scope silently.
```

## Final Verification Pass

```text
Review the integrated result after the implementation agents finish.

Check:
- obvious regressions
- acceptance criteria coverage
- missing tests
- architecture or layering violations

Do not modify code. Return only findings and residual risks.
```
