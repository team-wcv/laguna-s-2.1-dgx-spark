---
branch: chore/6a96dee3-laguna-august-profile
created: 2026-09-01
owner: codex-agent
status: active
scope: "Capture the verified August Laguna S 2.1 NVFP4 DGX Spark deployment profile and tuning evidence"
orchestraitor:
  ticket: 6a96dee3e6b6efd51fc86094
pr:
  url: https://github.com/team-wcv/laguna-s-2.1-dgx-spark/pull/1
  state: open
---

- Why this branch exists: Preserve the production-pinned August target/draft revisions and the measured single-Spark profile in the team-wcv fork.
- Changed paths: deployment install/preflight/serve/unit and endpoint defaults; current/historical docs; benchmark defaults; executable profile argument test.
- Validation run: all shell scripts parse; Python benchmark compiles; `tests/test-current-profile.sh` passes; fork preflight passes on the Spark against both pinned snapshots; live profile previously passed code, tool-call, reasoning, API, and Exo-runner health gates.
- Known follow-ups: The current Exo CUDA capability does not provide a vLLM worker engine; this recipe remains an OpenAI-compatible vLLM sidecar.
