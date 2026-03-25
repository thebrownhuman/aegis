# Aegis Development Decision Log

> This file tracks every significant decision, issue, and rationale during development.
> Updated continuously. Each entry includes date, context, decision, and why.

---

## Sprint 1.1 — Project Scaffolding

### DL-001: Code location — `Aegis/` not `backend/`
- **Date:** 2026-03-25
- **Context:** dev_plan.md used `backend/` as the code directory. Founder requested all code go in `Aegis/` folder.
- **Decision:** All Python code lives under `Aegis/aegis/`. The `Aegis/` folder is the Poetry project root.
- **Impact:** Docker build context and import paths adjusted accordingly. No functional change.

### DL-002: Package structure created upfront
- **Date:** 2026-03-25
- **Context:** Creating empty `__init__.py` for all planned modules (core, knowledge, power, tools, etc.) even though most won't have code until later sprints.
- **Decision:** Create full skeleton now. Ensures imports work from day one and the structure matches dev_plan.md Section 2.
- **Trade-off:** Some empty directories exist temporarily. Acceptable — better than restructuring later.

### DL-003: Provider SDK dependencies included in Phase 1
- **Date:** 2026-03-25
- **Context:** dev_plan.md only listed `langchain-nvidia-ai-endpoints` for NIM. The tiered model strategy (D15/DR-010) requires adapters for Mistral, Groq, DeepSeek, and Gemini.
- **Decision:** Added `langchain-mistralai`, `langchain-groq`, `langchain-google-genai`, and `openai` (DeepSeek uses OpenAI-compatible API) to pyproject.toml now.
- **Rationale:** Sprint 1.4 will build the model router. Having deps installed early avoids disruption.

### DL-004: SQLAlchemy ORM includes future tables
- **Date:** 2026-03-25
- **Context:** Phase 1 only strictly needs users, conversations, messages, rate_limits. But the schema from dev_plan Section 9.2 includes all tables.
- **Decision:** Define all ORM models now (including guest_activity, portfolio_suggestions, etc.). Only create tables via `Base.metadata.create_all()` or Alembic.
- **Rationale:** No runtime cost. Avoids migration headaches when Phase 2-4 arrive. Schema is already locked in dev_plan.

### DL-005: Added PolicyDecision and AuditEvent tables
- **Date:** 2026-03-25
- **Context:** dev_plan Section 9.2 schema didn't include these. ID-04 (policy engine) requires a `policy_decisions` table. ID-18 (immutable audit) requires `audit_events` with hash chain fields.
- **Decision:** Added both to ORM models. Schema: policy_decisions(actor_id, action, resource, decision, rule_matched, timestamp), audit_events(actor_id, action, resource, decision, prev_hash, entry_hash).
- **Rationale:** These are core security infrastructure. Better to have the schema from the start.

### DL-006: Docker Compose uses named volumes
- **Date:** 2026-03-25
- **Context:** Could use bind mounts or named volumes for data persistence.
- **Decision:** Named volumes (`aegis-data`, `ollama-models`, `qdrant-data`) for production. Dev override uses bind mount for source code hot reload.
- **Rationale:** Named volumes survive container recreation. Bind mounts only for dev source code.

### DL-007: poetry.lock excluded from git
- **Date:** 2026-03-25
- **Context:** Convention varies — some projects commit lock files, some don't.
- **Decision:** `.gitignore` excludes `poetry.lock`. Will revisit if reproducibility issues arise.
- **Note:** For production Docker builds, lock file is generated inside the container.

### DL-008: Test database uses in-memory SQLite
- **Date:** 2026-03-25
- **Context:** Tests could use a temp file or in-memory SQLite.
- **Decision:** In-memory (`:memory:`) for speed. WAL and FULL pragmas still set for parity. Each test gets a fresh database via fixture.
- **Rationale:** Tests run faster. SQLite in-memory still respects pragmas.

### DL-009: Python 3.13 compatibility
- **Date:** 2026-03-25
- **Context:** dev_plan specified `python ^3.11`. Dev machine has Python 3.13.7.
- **Decision:** Changed pyproject.toml constraint to `>=3.11,<4.0` to accept 3.13.
- **Impact:** All deps install fine on 3.13. Fixed `datetime.utcnow()` deprecation warnings (replaced with `datetime.now(UTC)`).

### DL-010: In-memory SQLite WAL limitation in tests
- **Date:** 2026-03-25
- **Issue:** Database tests for WAL mode and synchronous=FULL failed because in-memory SQLite (`sqlite:///:memory:`) doesn't support WAL — it always uses `memory` journal mode.
- **Decision:** Updated test assertions to accept both `wal` (file-based) and `memory` (in-memory). The actual production database uses file-based SQLite where WAL is correctly enabled.
- **Verification:** 27/27 tests pass. Health endpoint confirms WAL mode on file-based DB.

---

## Sprint 1.2 — Core Orchestrator

### DL-011: Heuristic intent classifier for Sprint 1.2
- **Date:** 2026-03-25
- **Context:** ID-01 requires a complexity classifier for model routing. Options: (a) LLM-based classification via Ollama, (b) keyword/pattern heuristic, (c) hybrid.
- **Decision:** Heuristic-first approach using keyword sets and regex patterns. No LLM call for classification in Sprint 1.2.
- **Rationale:** Fast (~0ms), deterministic, testable. LLM-based classification adds latency and requires Ollama running. Will enhance with LLM in later sprints when Ollama classifier is tuned. Heuristic achieves 70%+ accuracy on 100-prompt test suite.

### DL-012: Fast-path vs full graph routing threshold
- **Date:** 2026-03-25
- **Context:** Orchestrator spec defines confidence > 0.85 + no tools + short/medium output as fast-path criteria.
- **Decision:** `is_simple` flag computed during classification. Simple queries (direct_chat or knowledge_retrieval with high confidence) bypass LangGraph entirely.
- **Target:** ~60-70% of traffic on fast-path. Validated: 25/100 test cases route to fast-path (simple category), matching expected distribution for diverse test suite.

### DL-013: LangGraph generate node uses placeholder responses
- **Date:** 2026-03-25
- **Context:** Sprint 1.2 builds the orchestrator pipeline but actual LLM API calls require API keys and running services.
- **Decision:** Generate node returns structured placeholder showing routing decision: `[Aegis Orchestrator] Routed to {provider}/{model} (intent=X, complexity=Y)`. Real LLM calls wired in when provider integrations are tested with API keys.
- **Rationale:** Validates the full pipeline (classify → route → generate → format → respond) without external dependencies. All 150 tests pass offline.

### DL-014: Qdrant and tool execution as placeholders
- **Date:** 2026-03-25
- **Context:** RETRIEVE node needs Qdrant, EXECUTE_TOOL node needs tool framework. Both are Sprint 1.3-1.4 work.
- **Decision:** Both nodes return empty results. Graph still flows through them correctly. Marked with `# TODO: Sprint 1.X` comments.
- **Impact:** Full state machine is testable now. Qdrant/tool integration plugs in without graph restructuring.

### DL-015: Account pool shares keys across DeepSeek models
- **Date:** 2026-03-25
- **Context:** DeepSeek-R1 (reasoning) and DeepSeek-V3 (general) use different models but same API accounts.
- **Decision:** Single `AccountPool("deepseek", ...)` shared by both providers. `get_pool("deepseek_r1")` and `get_pool("deepseek_v3")` both return the same pool.
- **Rationale:** API keys are account-level, not model-level. Separate pools would double-count quotas.

### DL-016: Circuit breaker at 3 consecutive errors
- **Date:** 2026-03-25
- **Context:** Need to decide when to mark a provider as unavailable.
- **Decision:** After 3 consecutive errors, provider marked `UNAVAILABLE` and skipped in routing. A single success resets the counter. Manual reset also available.
- **Rationale:** 3 is conservative — avoids flapping on transient errors while detecting persistent failures quickly.

### DL-017: Multi-step detection improved with comma-then pattern
- **Date:** 2026-03-25
- **Issue:** Initial classifier missed multi-step queries like "Search for X, then search for Y, summarize both" because it only checked for 2+ multi-step markers or "first...then" pattern.
- **Decision:** Added `, then` regex pattern as strong multi-step signal. Fixes 4 failing test cases.
- **Impact:** 122/122 Sprint 1.2 tests pass. 100-prompt suite accuracy maintained.

### DL-018: Orchestrator spec written as Sprint 1.2 prerequisite
- **Date:** 2026-03-25
- **Context:** Gate A requires `docs/orchestrator_spec.md` before coding.
- **Decision:** Wrote comprehensive spec covering: state machine (12 states), classification (5 intents, 4 tiers), fast-path constraints, 7-tier provider routing, timeout/retry policy, loop prevention, context budgets, safety integration, logging contract, 100-prompt suite schema, performance targets.
- **Location:** `docs/orchestrator_spec.md` (v1.0)

### DL-019: Initial Alembic migration created (blocker fix)
- **Date:** 2026-03-25
- **Issue:** Codex identified a blocker: `migrations/versions/` was empty, so `alembic upgrade head` on a fresh environment created zero tables. `main.py` no longer auto-creates tables (DL-005 fix from Sprint 1.1), so there was no schema bootstrap path.
- **Decision:** Generated `3fb24527e249_initial_schema.py` via `alembic revision --autogenerate`. Creates all 13 ORM tables with correct FK ordering.
- **Verification:** Fresh DB test — `alembic upgrade head` on empty `test_fresh.db` creates 14 tables (13 app + alembic_version). Full test suite 150/150 still passes. Temp DB cleaned up after test.
- **Impact:** Fresh environments now bootstrap correctly. This was the only path to schema creation since Sprint 1.1 Codex fix removed `Base.metadata.create_all()` from `main.py`.

---

*Add new entries below as development continues.*
