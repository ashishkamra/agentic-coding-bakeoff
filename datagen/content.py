"""Content templates for coding-agent benchmark traces.

Generates realistic system prompts, tool schemas, repository maps, code files,
bug reports, patches, and tool outputs at specified token counts.
"""

import json
import random
from datagen.tokens import (
    estimate_tokens,
    pad_to_tokens,
    truncate_to_tokens,
    generate_random_code,
    generate_prose,
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert software engineer assistant. You help developers debug, fix, \
and improve code in large repositories. You have access to tools for reading files, \
searching code, running tests, applying patches, and executing terminal commands.

When asked to fix a bug:
1. Analyze the reported issue and identify the likely root cause.
2. Inspect relevant source files, tests, and logs using the provided tools.
3. Generate a minimal, correct patch that fixes the issue without introducing regressions.
4. Verify the fix by requesting test execution.
5. If tests fail, iterate on the patch until all tests pass.

Follow these guidelines:
- Prefer minimal patches over large refactors unless specifically asked.
- Always explain your reasoning before making changes.
- When generating patches, use unified diff format.
- When calling tools, provide complete arguments — do not omit required fields.
- If you need more context, ask for it rather than guessing.
- Consider edge cases: null inputs, empty collections, boundary values, concurrent access.
- Respect existing code style and conventions in the repository.

You are working in a monorepo that uses Python 3.12+ with type annotations, pytest for testing, \
SQLAlchemy for ORM, FastAPI for HTTP endpoints, and Redis for caching. The repository follows \
trunk-based development with feature flags for gradual rollouts.
"""


def generate_system_prompt(tokens: int = 2000, rng: random.Random | None = None) -> str:
    return pad_to_tokens(_SYSTEM_PROMPT_TEMPLATE, tokens, rng)


# ---------------------------------------------------------------------------
# Tool / MCP schemas
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path. Returns the file content as a string with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to read"},
                    "start_line": {"type": "integer", "description": "First line to read (1-based, inclusive)"},
                    "end_line": {"type": "integer", "description": "Last line to read (1-based, inclusive)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if it does not exist or overwriting if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "The full content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a patch to one or more files. The patch should be in unified diff format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path relative to repo root"},
                                "diff": {"type": "string", "description": "Unified diff content for this file"},
                            },
                            "required": ["path", "diff"],
                        },
                        "description": "List of file patches to apply",
                    },
                },
                "required": ["files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": "Execute a shell command in the repository root directory and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                    "working_dir": {"type": "string", "description": "Working directory relative to repo root"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a pattern across files in the repository using ripgrep-style matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in"},
                    "include": {"type": "string", "description": "Glob pattern for files to include (e.g. '*.py')"},
                    "exclude": {"type": "string", "description": "Glob pattern for files to exclude"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a given path, optionally with recursion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                    "recursive": {"type": "boolean", "description": "Whether to list recursively"},
                    "pattern": {"type": "string", "description": "Glob pattern to filter results"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite or specific test files using pytest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_path": {"type": "string", "description": "Path to test file or directory"},
                    "marker": {"type": "string", "description": "pytest marker to select tests (e.g. 'unit', 'integration')"},
                    "keyword": {"type": "string", "description": "pytest -k expression to filter tests"},
                    "verbose": {"type": "boolean", "description": "Enable verbose output"},
                    "fail_fast": {"type": "boolean", "description": "Stop on first failure"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_diagnostics",
            "description": "Get linting diagnostics, type checking errors, or static analysis results for files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or directory to analyze"},
                    "tool": {"type": "string", "enum": ["mypy", "ruff", "pylint", "pyright"], "description": "Analysis tool to use"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show the git diff for staged or unstaged changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Limit diff to this path"},
                    "staged": {"type": "boolean", "description": "Show staged changes (default: unstaged)"},
                    "commit": {"type": "string", "description": "Compare against this commit ref"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent git commit history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Filter log to this path"},
                    "max_count": {"type": "integer", "description": "Maximum commits to show"},
                    "since": {"type": "string", "description": "Show commits after this date"},
                    "author": {"type": "string", "description": "Filter by author name or email"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_symbol_info",
            "description": "Get type information, definition location, and references for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The symbol name to look up"},
                    "file": {"type": "string", "description": "File where the symbol is used"},
                    "line": {"type": "integer", "description": "Line number where the symbol appears"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "Execute a read-only SQL query against the development database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query to execute"},
                    "database": {"type": "string", "description": "Database name (default: main)"},
                    "limit": {"type": "integer", "description": "Row limit (default: 100)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_logs",
            "description": "Search application logs for patterns in the specified time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name to search logs for"},
                    "pattern": {"type": "string", "description": "Pattern to search for in logs"},
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "since": {"type": "string", "description": "Start time (ISO 8601)"},
                    "until": {"type": "string", "description": "End time (ISO 8601)"},
                    "limit": {"type": "integer", "description": "Maximum log entries to return"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics",
            "description": "Query Prometheus metrics for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PromQL query"},
                    "start": {"type": "string", "description": "Start time"},
                    "end": {"type": "string", "description": "End time"},
                    "step": {"type": "string", "description": "Query resolution step (e.g. '1m')"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_branch",
            "description": "Create a new git branch from the current HEAD or a specified ref.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Branch name"},
                    "from_ref": {"type": "string", "description": "Base ref (default: HEAD)"},
                },
                "required": ["name"],
            },
        },
    },
]


def generate_tool_schemas(tokens: int = 8000, rng: random.Random | None = None) -> tuple[str, list[dict]]:
    """Returns (text_representation, schema_list) where text is for embedding in messages."""
    schemas_json = json.dumps(_TOOL_SCHEMAS, indent=2, ensure_ascii=False)
    text = f"# Available Tools\n\nYou have access to the following tools:\n\n```json\n{schemas_json}\n```\n"
    text = pad_to_tokens(text, tokens, rng)
    return text, _TOOL_SCHEMAS


# ---------------------------------------------------------------------------
# Repository map
# ---------------------------------------------------------------------------

_REPO_MODULES = [
    ("src/api", "FastAPI HTTP endpoint handlers", ["routes.py", "middleware.py", "dependencies.py", "schemas.py", "responses.py"]),
    ("src/auth", "Authentication and authorization", ["jwt_handler.py", "oauth.py", "permissions.py", "session.py", "mfa.py"]),
    ("src/billing", "Billing, invoicing, and payment processing", ["stripe_client.py", "invoices.py", "subscriptions.py", "webhooks.py", "plans.py"]),
    ("src/cache", "Redis caching layer", ["redis_client.py", "decorators.py", "invalidation.py", "serializers.py"]),
    ("src/core", "Core domain models and business logic", ["models.py", "services.py", "events.py", "exceptions.py", "constants.py"]),
    ("src/db", "Database access and migrations", ["engine.py", "session.py", "repositories.py", "migrations/", "seeds.py"]),
    ("src/email", "Email sending and templates", ["sender.py", "templates.py", "queue.py", "tracking.py"]),
    ("src/integrations", "Third-party service integrations", ["slack.py", "github.py", "jira.py", "pagerduty.py", "datadog.py"]),
    ("src/ml", "Machine learning pipeline", ["embeddings.py", "inference.py", "training.py", "features.py", "model_registry.py"]),
    ("src/search", "Full-text search engine", ["indexer.py", "query_builder.py", "analyzers.py", "synonyms.py"]),
    ("src/storage", "File storage abstraction", ["s3_client.py", "local.py", "interface.py", "presigned.py"]),
    ("src/tasks", "Background task processing", ["celery_app.py", "handlers.py", "schedules.py", "retry.py"]),
    ("src/telemetry", "Observability and tracing", ["tracer.py", "metrics.py", "logging.py", "exporters.py"]),
    ("src/workers", "Worker processes", ["consumer.py", "producer.py", "dlq.py", "scaling.py"]),
    ("tests", "Test suite", ["conftest.py", "test_api/", "test_auth/", "test_billing/", "test_core/", "test_db/"]),
]


def generate_repo_map(tokens: int, repo_name: str = "acme-platform", rng: random.Random | None = None) -> str:
    rng = rng or random.Random(42)
    lines = [
        f"# Repository: {repo_name}",
        "",
        "## Project Structure",
        "",
    ]

    for module_path, description, files in _REPO_MODULES:
        lines.append(f"### `{module_path}/` — {description}")
        lines.append("")
        for f in files:
            if f.endswith("/"):
                lines.append(f"  - `{f}` (directory)")
            else:
                loc = rng.randint(50, 800)
                lines.append(f"  - `{f}` ({loc} lines)")
        lines.append("")

        summary_tokens = rng.randint(200, 600)
        summary = generate_prose(summary_tokens, rng)
        lines.append(f"**Module summary**: {summary}")
        lines.append("")

    text = "\n".join(lines)
    return pad_to_tokens(text, tokens, rng)


# ---------------------------------------------------------------------------
# Code files (active context)
# ---------------------------------------------------------------------------

_FILE_TEMPLATES = {
    "api_handler": '''\
"""API endpoint handlers for the {module} module."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from pydantic import BaseModel, Field

from src.db.session import get_db
from src.auth.permissions import require_permission
from src.core.models import {model_class}
from src.core.services import {service_class}
from src.cache.decorators import cached
from src.telemetry.tracer import trace_span

router = APIRouter(prefix="/{module}", tags=["{module}"])

class {model_class}Create(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    config: dict = Field(default_factory=dict)
    is_active: bool = True

class {model_class}Response(BaseModel):
    id: int
    name: str
    description: Optional[str]
    config: dict
    is_active: bool
    created_at: str
    updated_at: str

    model_config = {{"from_attributes": True}}

class {model_class}List(BaseModel):
    items: List[{model_class}Response]
    total: int
    page: int
    page_size: int

''',
    "service": '''\
"""Business logic for the {module} module."""

import logging
from typing import Optional, List
from datetime import datetime, timezone

from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from src.core.models import {model_class}
from src.core.events import publish_event
from src.core.exceptions import NotFoundError, ConflictError, ValidationError
from src.cache.invalidation import invalidate_pattern
from src.telemetry.tracer import trace_span

logger = logging.getLogger(__name__)

class {service_class}:
    def __init__(self, db: AsyncSession):
        self.db = db

    @trace_span("get_{module}")
    async def get_by_id(self, id: int) -> Optional[{model_class}]:
        result = await self.db.execute(
            select({model_class}).where({model_class}.id == id)
        )
        return result.scalar_one_or_none()

    @trace_span("list_{module}")
    async def list(
        self,
        page: int = 1,
        page_size: int = 20,
        is_active: Optional[bool] = None,
        search: Optional[str] = None,
    ) -> tuple[List[{model_class}], int]:
        query = select({model_class})
        count_query = select(func.count()).select_from({model_class})

        if is_active is not None:
            query = query.where({model_class}.is_active == is_active)
            count_query = count_query.where({model_class}.is_active == is_active)

        if search:
            query = query.where({model_class}.name.ilike(f"%{{search}}%"))
            count_query = count_query.where({model_class}.name.ilike(f"%{{search}}%"))

        total = (await self.db.execute(count_query)).scalar() or 0
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self.db.execute(query)
        return list(result.scalars().all()), total

''',
    "test": '''\
"""Tests for the {module} module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.models import {model_class}
from src.core.services import {service_class}

@pytest.fixture
def mock_db():
    db = AsyncMock(spec=AsyncSession)
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock()
    return db

@pytest.fixture
def service(mock_db):
    return {service_class}(mock_db)

@pytest.fixture
def sample_{module}():
    return {model_class}(
        id=1,
        name="test-{module}",
        description="Test description",
        config={{}},
        is_active=True,
    )

class TestGet{model_class}:
    @pytest.mark.asyncio
    async def test_get_existing(self, service, mock_db, sample_{module}):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_{module}
        mock_db.execute.return_value = mock_result
        result = await service.get_by_id(1)
        assert result is not None
        assert result.id == 1
        assert result.name == "test-{module}"

    @pytest.mark.asyncio
    async def test_get_missing(self, service, mock_db):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        result = await service.get_by_id(999)
        assert result is None

''',
}

_MODULE_NAMES = [
    ("billing", "Invoice", "InvoiceService"),
    ("auth", "User", "UserService"),
    ("search", "SearchIndex", "SearchService"),
    ("cache", "CacheEntry", "CacheService"),
    ("email", "EmailTemplate", "EmailService"),
    ("tasks", "BackgroundJob", "JobService"),
    ("storage", "StoredFile", "StorageService"),
    ("integrations", "WebhookConfig", "WebhookService"),
    ("ml", "ModelVersion", "MLService"),
    ("telemetry", "MetricSeries", "MetricService"),
]


def generate_code_files(tokens: int, module_index: int = 0, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(42)
    module, model_class, service_class = _MODULE_NAMES[module_index % len(_MODULE_NAMES)]

    parts = []
    for template_name, template in _FILE_TEMPLATES.items():
        code = template.format(module=module, model_class=model_class, service_class=service_class)
        filename = f"src/{module}/{template_name}.py"
        parts.append(f"# File: {filename}\n\n{code}")

    text = "\n\n---\n\n".join(parts)

    if estimate_tokens(text) < tokens:
        extra_code = generate_random_code(tokens - estimate_tokens(text), rng)
        text += f"\n\n# File: src/{module}/utils.py\n\n{extra_code}"

    return truncate_to_tokens(text, tokens) if estimate_tokens(text) > tokens else pad_to_tokens(text, tokens, rng)


# ---------------------------------------------------------------------------
# Bug reports
# ---------------------------------------------------------------------------

_BUG_TEMPLATES = [
    {
        "title": "NullPointerException in {module} service when processing empty collections",
        "body": "When the {module} service receives an empty list of items, it crashes with a NullPointerException "
                "at line {line} in {file}. The root cause appears to be a missing null check before accessing "
                "the first element of the collection. This affects the {endpoint} endpoint and causes 500 errors "
                "for approximately 2% of requests in production. Steps to reproduce: 1) Send a POST request to "
                "/{module}/batch with an empty items array. 2) Observe the 500 Internal Server Error response. "
                "Expected: The endpoint should return a 200 with an empty results array.",
    },
    {
        "title": "Race condition in {module} cache invalidation causes stale reads",
        "body": "Under high concurrency, the {module} cache invalidation does not properly synchronize with "
                "writes, causing subsequent reads to return stale data. The issue is in {file} around line {line} "
                "where the cache delete and database write are not wrapped in a transaction. This manifests as "
                "users seeing old data for 5-30 seconds after an update. The {module} team reported this after "
                "observing inconsistencies during peak traffic.",
    },
    {
        "title": "Memory leak in {module} background worker due to unclosed database connections",
        "body": "The {module} background worker process gradually consumes more memory over time, eventually "
                "triggering OOM kills after about 12 hours of operation. Investigation shows that database "
                "connections are being acquired in {file} at line {line} but not properly released when the "
                "task fails with an exception. The connection pool grows unbounded. This blocks the {endpoint} "
                "queue and causes task processing delays.",
    },
    {
        "title": "SQL injection vulnerability in {module} search endpoint",
        "body": "The {module} search endpoint at /{module}/search does not properly sanitize the 'query' "
                "parameter before interpolating it into a SQL query in {file} line {line}. An attacker can "
                "craft a malicious query parameter to extract data from other tables. Priority: CRITICAL. "
                "The fix should use parameterized queries instead of string interpolation.",
    },
    {
        "title": "Timeout handling regression in {module} API client",
        "body": "After the recent refactor of the HTTP client in {file}, the timeout parameter at line {line} "
                "is no longer being passed to the underlying requests library. This causes the {module} "
                "integration to hang indefinitely when the upstream service is unresponsive, instead of failing "
                "after the configured 30-second timeout. The {endpoint} endpoint now shows p99 latency of 120s "
                "instead of the expected 35s.",
    },
    {
        "title": "Incorrect pagination in {module} list endpoint skips records",
        "body": "The {module} list endpoint at /{module}/ returns incorrect results when paginating through "
                "large datasets. Records are being skipped because the offset calculation in {file} at line "
                "{line} uses 0-based page indexing but the frontend sends 1-based page numbers. This causes "
                "every page after the first to skip page_size records. Reported by the frontend team after "
                "users noticed missing items in the {module} listing.",
    },
    {
        "title": "Deadlock in {module} batch processing with concurrent transactions",
        "body": "When multiple batch operations run concurrently in the {module} service, a deadlock occurs "
                "between the batch update and the event publishing transaction. The issue is in {file} around "
                "line {line} where two transactions acquire locks in different orders. This causes the batch "
                "endpoint /{module}/batch to return 500 errors under load. The deadlock is detectable in the "
                "database slow query logs.",
    },
    {
        "title": "Authentication bypass in {module} webhook handler",
        "body": "The webhook handler for {module} at /{module}/webhooks does not verify the webhook signature "
                "in {file} line {line}. While the signature verification function exists, it is called but its "
                "return value is not checked. An attacker can send forged webhook payloads to trigger actions "
                "in the {module} system. Priority: HIGH. Related endpoint: {endpoint}.",
    },
    {
        "title": "Data corruption in {module} migration script",
        "body": "The latest migration for the {module} module incorrectly converts existing data during the "
                "schema change. In {file} at line {line}, the migration assumes all existing records have a "
                "non-null 'config' field, but records created before v2.1 have NULL configs. This causes the "
                "migration to fail partway through, leaving the database in an inconsistent state. Affects "
                "the {endpoint} endpoint which reads from the migrated table.",
    },
    {
        "title": "Rate limiter not applied to {module} bulk endpoints",
        "body": "The rate limiter middleware skips the {module} bulk endpoints because they use a different "
                "URL pattern than expected. In {file} at line {line}, the rate limiter regex matches "
                "'/{module}/{{id}}' but not '/{module}/bulk'. This allows unlimited requests to the bulk "
                "endpoints, enabling potential DoS attacks. The {endpoint} handler processes large payloads "
                "without any request throttling.",
    },
]


def generate_bug_report(issue_id: int, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(issue_id)
    template = _BUG_TEMPLATES[issue_id % len(_BUG_TEMPLATES)]
    module, model_class, service_class = _MODULE_NAMES[issue_id % len(_MODULE_NAMES)]

    files = [f"src/{module}/services.py", f"src/{module}/routes.py", f"src/{module}/handlers.py"]
    endpoints = [f"/{module}/", f"/{module}/{{id}}", f"/{module}/batch", f"/{module}/search"]

    title = template["title"].format(module=module)
    body = template["body"].format(
        module=module,
        file=rng.choice(files),
        line=rng.randint(40, 350),
        endpoint=rng.choice(endpoints),
    )

    return f"## Bug Report: {title}\n\n**Issue ID**: ACME-{1000 + issue_id}\n**Priority**: {'CRITICAL' if 'security' in body.lower() or 'injection' in body.lower() else 'HIGH'}\n**Component**: {module}\n\n{body}"


# ---------------------------------------------------------------------------
# Tool outputs
# ---------------------------------------------------------------------------

def generate_test_output(passing: bool, tokens: int, rng: random.Random) -> str:
    if passing:
        lines = [
            "============================= test session starts ==============================",
            "platform linux -- Python 3.12.4, pytest-8.2.0, pluggy-1.5.0",
            "rootdir: /workspace/acme-platform",
            "configfile: pyproject.toml",
            "plugins: asyncio-0.23.7, cov-5.0.0, anyio-4.4.0",
            "collected 47 items",
            "",
        ]
        for i in range(rng.randint(8, 20)):
            test_name = f"test_{rng.choice(['get', 'create', 'update', 'delete', 'list', 'search', 'validate', 'process'])}_{rng.choice(['basic', 'edge_case', 'empty', 'large', 'concurrent', 'timeout', 'error'])}"
            lines.append(f"tests/test_{rng.choice(['api', 'service', 'model', 'utils'])}.py::{test_name} PASSED")
        lines.extend([
            "",
            f"============================== {len(lines) - 6} passed in 3.42s ===============================",
        ])
    else:
        module = rng.choice([m[0] for m in _MODULE_NAMES])
        lines = [
            "============================= test session starts ==============================",
            "platform linux -- Python 3.12.4, pytest-8.2.0, pluggy-1.5.0",
            "rootdir: /workspace/acme-platform",
            "",
        ]
        for i in range(rng.randint(3, 8)):
            lines.append(f"tests/test_{module}.py::test_{'_'.join(rng.choices(['get', 'create', 'validate', 'process', 'basic', 'error'], k=2))} PASSED")

        lines.extend([
            f"tests/test_{module}.py::test_{'_'.join(rng.choices(['handle', 'check', 'verify', 'process'], k=2))} FAILED",
            "",
            "=================================== FAILURES ===================================",
            f"_________________________________ test_{'_'.join(rng.choices(['handle', 'check'], k=2))} __________________________________",
            "",
            f"    def test_failing():",
            f"        result = service.process(input_data)",
            f">       assert result.status == 'completed'",
            f"E       AssertionError: assert 'error' == 'completed'",
            f"E         - error",
            f"E         + completed",
            "",
            f"tests/test_{module}.py:142: AssertionError",
            "",
            "=========================== short test summary info ============================",
            f"FAILED tests/test_{module}.py::test_failing - AssertionError",
            f"========================= 1 failed, {rng.randint(5, 15)} passed in 4.21s =========================",
        ])

    text = "\n".join(lines)
    return pad_to_tokens(text, tokens, rng) if estimate_tokens(text) < tokens else truncate_to_tokens(text, tokens)


def generate_grep_output(tokens: int, rng: random.Random) -> str:
    lines = []
    module = rng.choice([m[0] for m in _MODULE_NAMES])
    pattern = rng.choice(["handle_error", "validate_input", "process_request", "cache_key", "serialize"])
    for i in range(rng.randint(5, 25)):
        file = f"src/{module}/{rng.choice(['services', 'routes', 'handlers', 'utils', 'middleware'])}.py"
        line_num = rng.randint(10, 500)
        indent = "    " * rng.randint(0, 3)
        context = f"{indent}{pattern}({', '.join(rng.choices(['self', 'request', 'data', 'config', 'session', 'user'], k=rng.randint(1, 4)))})"
        lines.append(f"{file}:{line_num}: {context}")

    text = "\n".join(lines)
    return pad_to_tokens(text, tokens, rng) if estimate_tokens(text) < tokens else truncate_to_tokens(text, tokens)


def generate_tool_output(kind: str, tokens: int, rng: random.Random) -> str:
    if kind == "test_pass":
        return generate_test_output(True, tokens, rng)
    elif kind == "test_fail":
        return generate_test_output(False, tokens, rng)
    elif kind == "grep":
        return generate_grep_output(tokens, rng)
    elif kind == "patch_applied":
        return pad_to_tokens("Patch applied successfully to 1 file(s).\nModified: src/{}/services.py (+12, -3 lines)".format(
            rng.choice([m[0] for m in _MODULE_NAMES])
        ), tokens, rng)
    elif kind == "diagnostics":
        lines = ["Running mypy on src/..."]
        for i in range(rng.randint(2, 8)):
            file = f"src/{rng.choice([m[0] for m in _MODULE_NAMES])}/{rng.choice(['services', 'routes'])}.py"
            lines.append(f"{file}:{rng.randint(10, 300)}: error: {rng.choice(['Incompatible types', 'Missing return statement', 'Argument has incompatible type', 'Name is not defined'])}")
        lines.append(f"Found {len(lines) - 1} errors in {rng.randint(2, 5)} files")
        text = "\n".join(lines)
        return pad_to_tokens(text, tokens, rng) if estimate_tokens(text) < tokens else truncate_to_tokens(text, tokens)
    else:
        return generate_prose(tokens, rng)


# ---------------------------------------------------------------------------
# Patches & structured tool call output
# ---------------------------------------------------------------------------

def generate_patch(issue_id: int, tokens: int, rng: random.Random) -> str:
    module, _, _ = _MODULE_NAMES[issue_id % len(_MODULE_NAMES)]
    file = f"src/{module}/services.py"

    lines = [
        f"Based on my analysis, here is the fix for ACME-{1000 + issue_id}:",
        "",
        f"The issue is in `{file}`. The fix adds proper validation and error handling:",
        "",
        f"```diff",
        f"--- a/{file}",
        f"+++ b/{file}",
        f"@@ -{rng.randint(40, 200)},{rng.randint(5, 15)} +{rng.randint(40, 200)},{rng.randint(8, 20)} @@",
    ]

    for _ in range(rng.randint(3, 8)):
        lines.append(f" {' ' * rng.randint(0, 3)}{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz_ .()=:', k=rng.randint(15, 60)))}")
        lines.append(f"-{' ' * rng.randint(0, 3)}{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz_ .()=:', k=rng.randint(15, 60)))}")
        lines.append(f"+{' ' * rng.randint(0, 3)}{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz_ .()=:', k=rng.randint(15, 60)))}")
        lines.append(f"+{' ' * rng.randint(0, 3)}{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz_ .()=:', k=rng.randint(15, 60)))}")

    lines.append("```")
    lines.append("")
    lines.append("This change ensures proper input validation before processing.")

    text = "\n".join(lines)
    return pad_to_tokens(text, tokens, rng) if estimate_tokens(text) < tokens else truncate_to_tokens(text, tokens)


def generate_structured_patch_output(issue_id: int, rng: random.Random) -> tuple[str, list[dict]]:
    """Returns (text_for_matching, output_messages_list) for a tool call output."""
    module, _, _ = _MODULE_NAMES[issue_id % len(_MODULE_NAMES)]
    file_path = f"src/{module}/services.py"

    diff_content = f"@@ -{rng.randint(40, 200)},8 +{rng.randint(40, 200)},12 @@\n"
    for _ in range(rng.randint(3, 6)):
        diff_content += f" {''.join(rng.choices('abcdefghijklmnopqrstuvwxyz_ .()=:', k=rng.randint(20, 50)))}\n"
        diff_content += f"-{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz_ .()=:', k=rng.randint(20, 50)))}\n"
        diff_content += f"+{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz_ .()=:', k=rng.randint(20, 50)))}\n"

    arguments = json.dumps({
        "files": [{"path": file_path, "diff": diff_content}]
    })

    tool_call_id = f"call_{rng.randint(100000, 999999)}"

    output_messages = [{
        "role": "assistant",
        "parts": [{
            "type": "tool_call",
            "id": tool_call_id,
            "name": "apply_patch",
            "arguments": arguments,
        }]
    }]

    text = f"tool_call: apply_patch(files=[{{path: {file_path}, diff: ...}}])"

    return text, output_messages, tool_call_id


# ---------------------------------------------------------------------------
# Corporate coding guidelines
# ---------------------------------------------------------------------------

def generate_coding_guidelines(tokens: int = 5000, rng: random.Random | None = None) -> str:
    template = """\
# ACME Platform — Engineering Standards & Coding Guidelines

## Python Style
- Use Python 3.12+ features including type annotations on all public functions.
- Follow PEP 8 with a line length limit of 120 characters.
- Use `ruff` for linting and `black` for formatting (configured in pyproject.toml).
- Prefer f-strings over `.format()` or `%` formatting.
- Use `pathlib.Path` instead of `os.path` for filesystem operations.

## Error Handling
- Never catch bare `Exception` — always catch specific exception types.
- Use custom exception classes defined in `src/core/exceptions.py`.
- Log exceptions with `logger.exception()` to capture stack traces.
- Return structured error responses from API endpoints, not raw exceptions.

## Database
- All queries must use parameterized statements — never interpolate user input into SQL.
- Use async SQLAlchemy sessions (`AsyncSession`) for all database operations.
- Always acquire sessions via the `get_db` dependency, never create sessions manually.
- Wrap related writes in explicit transactions with proper rollback on failure.
- Add database indexes for any column used in WHERE or JOIN clauses.

## Testing
- All new code must have unit tests with >=80% line coverage.
- Integration tests use a real PostgreSQL test database, not mocks.
- Use `pytest.mark.asyncio` for async test functions.
- Fixtures go in `conftest.py` at the appropriate directory level.
- Test file names must match `test_*.py` and test functions must start with `test_`.

## API Design
- Use RESTful conventions: GET for reads, POST for creates, PUT for full updates, PATCH for partial updates.
- All endpoints must validate input using Pydantic models.
- Use HTTP status codes correctly: 201 for creates, 204 for deletes, 422 for validation errors.
- Paginated list endpoints must return `{items, total, page, page_size}`.

## Security
- Never log sensitive data (passwords, tokens, PII).
- All webhook endpoints must verify request signatures.
- Use constant-time comparison for secret comparison (`hmac.compare_digest`).
- Rate limit all public endpoints. Configure limits in `src/api/middleware.py`.

## Performance
- Use Redis caching for frequently read, rarely written data.
- Cache keys must include a version prefix for safe invalidation.
- Background tasks that take >100ms should use Celery, not inline processing.
- Database queries in request handlers must complete within the 5-second query timeout.
"""
    return pad_to_tokens(template, tokens, rng)


# ---------------------------------------------------------------------------
# API documentation
# ---------------------------------------------------------------------------

def generate_api_docs(tokens: int, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(42)
    template = """\
# ACME Platform API Reference

## Authentication
All API requests require a valid JWT bearer token in the Authorization header:
```
Authorization: Bearer <token>
```
Tokens are issued via POST /auth/login and expire after 24 hours. Use POST /auth/refresh to obtain a new token.

## Common Response Formats

### Success Response
```json
{
    "data": { ... },
    "meta": {
        "request_id": "uuid",
        "timestamp": "ISO8601"
    }
}
```

### Error Response
```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Human readable message",
        "details": [ ... ]
    }
}
```

## Rate Limits
- Standard endpoints: 100 req/min per API key
- Bulk endpoints: 10 req/min per API key
- Search endpoints: 30 req/min per API key

Rate limit headers are included in all responses:
- X-RateLimit-Limit
- X-RateLimit-Remaining
- X-RateLimit-Reset

## Endpoints

"""
    for module, model_class, _ in _MODULE_NAMES[:8]:
        template += f"""### {model_class} API

#### GET /{module}/
List all {module} resources with pagination.

Query Parameters:
- `page` (int, default=1): Page number
- `page_size` (int, default=20, max=100): Items per page
- `search` (string): Full-text search query
- `is_active` (bool): Filter by active status
- `sort` (string): Sort field (created_at, name, updated_at)
- `order` (string): Sort order (asc, desc)

Response: {model_class}List

#### GET /{module}/{{id}}
Get a single {module} resource by ID.

Response: {model_class}Response

#### POST /{module}/
Create a new {module} resource.

Request Body: {model_class}Create
Response: {model_class}Response (201 Created)

#### PUT /{module}/{{id}}
Update an existing {module} resource.

Request Body: {model_class}Update
Response: {model_class}Response

#### DELETE /{module}/{{id}}
Delete a {module} resource.

Response: 204 No Content

"""
    return pad_to_tokens(template, tokens, rng)
