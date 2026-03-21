# CLAUDE.md — polyfit-image-compress

## Project Overview

Image compression library using Least Squares Polynomial Surface Fitting with quantum computing module (PennyLane).

- **Language**: Python 3.10+
- **Package**: `src/polyfit_compress/`
- **Tests**: `tests/` (pytest)
- **Docs**: `DOCS/`

## Rules

### Code Quality — Mandatory Before Every Commit

Run these commands BEFORE every commit. If any fails, fix the issues before committing.

```bash
# Lint (like flutter analyze)
ruff check src/ tests/

# Auto-fix lint issues (like dart fix --apply)
ruff check --fix src/ tests/

# Format (like flutter format)
ruff format src/ tests/

# Type checking
mypy src/polyfit_compress/
```

Order: `ruff check --fix` -> `ruff format` -> `mypy` -> then commit.

If pre-commit hooks are installed (`pre-commit install`), ruff and mypy run automatically on `git commit`. NEVER skip hooks with `--no-verify`.

### Branching Strategy — 3-Level Structure

ALL development work MUST follow this branching strategy:

```
master (production-ready, always deployable)
  └── feature/{name}                         <- Safety branch (merges to master)
        └── feature/{name}-setup             <- Integration branch (stubs, empty functions, imports — prevents merge conflicts)
              ├── {name}/{sub-feature-1}     <- Sub-feature branch (actual implementation)
              ├── {name}/{sub-feature-2}
              └── ...
```

#### Flow

1. **Sub-feature branches** -> PR -> merge into **integration branch** (`feature/{name}-setup`)
2. **Integration branch** -> PR -> merge into **safety branch** (`feature/{name}`)
3. **Safety branch** -> PR -> merge into **master**

#### Branch Naming

| Level | Format | Example |
|-------|--------|---------|
| Safety branch | `feature/{name}` | `feature/cli` |
| Integration branch | `feature/{name}-setup` | `feature/cli-setup` |
| Sub-feature | `{name}/{sub-feature}` | `cli/commands` |

#### Integration Branch Protocol

When creating the integration/setup branch:
1. Create ALL necessary files with minimal boilerplate (imports, class stubs, empty functions)
2. This prevents merge conflicts when sub-features are developed in parallel
3. Commit message: `feat({scope}): scaffold {feature-name} module structure`

### PR Workflow

1. **Before creating ANY PR**: STOP and ask the user "Can I create this PR?" with source branch, target branch, and list of commits. WAIT for approval.
2. **After PR is created**: Comment `@claude review this code` on the PR.
3. **Wait for Claude's review response**.
4. **Fix all issues** found in the review, commit the fixes.
5. **Comment** `@claude fixes done` on the PR.
6. **User evaluates and approves** the merge. NEVER merge yourself.

### Session Close Protocol

When the user says "vamos a cerrar sesion", "seguimos luego", "session close", or any variation:

1. **Update `DOCS/08-Sprint.md`** — Mark completed tasks, update progress tracker, reflect current phase status.
2. **Update any other DOCS/** file that has become outdated during the session.
3. **Save session summary to engram** via `mem_session_summary`.
4. Only THEN confirm the session is closed.

### Commit Convention

Format: `<type>(<scope>): <description>`

- **Types**: feat, fix, refactor, docs, test, bench, chore, ci
- **Scopes**: compressor, models, io, quantum, cli, metrics, viz, config, deps
- **Rules**: Imperative mood, max 72 chars, body explains WHY not WHAT
- **NEVER** add `Co-Authored-By` or AI attribution

### General Rules

- NEVER commit directly to master.
- NEVER force push (`git push --force`).
- NEVER skip pre-commit hooks (`--no-verify`).
- NEVER delete branches without user approval.
- NEVER build after changes.
- Use `bat/rg/fd/sd/eza` instead of `cat/grep/find/sed/ls`.
- When asking the user a question, STOP and wait for response.
