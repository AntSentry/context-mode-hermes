"""Clean-distribution smoke for the context-mode-hermes wheel.

Drives the nested real engine installed by the wheel's public
``context-mode-hermes --install-engine`` command through the engine's public
MCP stdio interface — the same transport production Hermes uses — and asserts
independent expected outcomes:

* ``ctx_execute`` really executes code (the engine must print a value it
  computed, not something the caller supplied);
* ``ctx_index`` stores content under a unique marker and ``ctx_search``
  retrieves text that was stored — an echo of the query cannot satisfy it.

The script also asserts that ``context_mode_hermes`` imports from the virtual
environment that installed the wheel, not from a source checkout. Run it with
the smoke virtualenv's interpreter from a working directory outside the
checkout:

    smoke-venv/bin/python .github/scripts/clean_distribution_smoke.py --home <home>

Exits non-zero with a diagnostic on the first failed expectation. Requires the
``mcp`` client SDK (pinned to the same version as Hermes) in the same
environment.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
import time
from pathlib import Path

SMOKE_SOURCE = "clean-distribution-smoke"
STORED_SENTENCE = "The verification lantern drifts over cobalt water."

#: Environment variables another agent might set; never inherit their paths.
FOREIGN_ENV = (
    "CLAUDE_PLUGIN_ROOT", "CLAUDE_CONFIG_DIR", "CLAUDE_PROJECT_DIR",
    "CLAUDE_SESSION_ID", "CLAUDE_CODE_ENTRYPOINT", "PI_CODING_AGENT_DIR",
    "PI_PROJECT_DIR", "OMP_PLUGIN_ROOT", "CONTEXT_MODE_PLATFORM",
    "HERMES_HOME", "CONTEXT_MODE_DIR", "CONTEXT_MODE_BIN",
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_import_origin() -> None:
    import context_mode_hermes

    package = Path(context_mode_hermes.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    if not package.is_relative_to(prefix):
        fail(
            f"context_mode_hermes imported from {package}, which is outside the "
            f"smoke virtualenv {prefix}; the wheel did not provide the package"
        )
    print(f"ok: context_mode_hermes imported from the wheel install ({package})")


async def engine_round_trip(binary: Path) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    with tempfile.TemporaryDirectory(prefix="clean-smoke-") as directory:
        isolated = dict(os.environ)
        for name in FOREIGN_ENV:
            isolated.pop(name, None)
        isolated.update({
            "CONTEXT_MODE_DIR": str(Path(directory) / "data"),
            "CLAUDE_CONFIG_DIR": str(Path(directory) / "host"),
            "XDG_CONFIG_HOME": str(Path(directory) / "xdg"),
            "CLAUDE_PROJECT_DIR": directory,
            "CONTEXT_MODE_PROJECT_DIR": directory,
            "CLAUDE_SESSION_ID": "clean-distribution-smoke",
        })
        parameters = StdioServerParameters(command=str(binary), env=isolated, cwd=directory)
        async with asyncio.timeout(120):
            async with stdio_client(parameters) as (reader, writer):
                async with ClientSession(reader, writer) as client:
                    await client.initialize()

                    async def call(name: str, arguments: dict) -> str:
                        result = await client.call_tool(name, arguments=arguments)
                        text = "\n".join(
                            getattr(item, "text", "") for item in result.content
                        )
                        if result.is_error:
                            fail(f"engine tool {name} returned an error: {text or name}")
                        return text

                    executed = await call("ctx_execute", {
                        "language": "python",
                        "code": "print(sum([19, 23]))",
                    })
                    if not re.search(r"(?m)^42\s*$", executed):
                        fail(f"ctx_execute did not run Python (output: {executed!r})")
                    print("ok: ctx_execute computed 42 with the real engine")

                    marker = f"cleansmoke{time.time_ns()}"
                    await call("ctx_index", {
                        "content": f"# {marker}\n{STORED_SENTENCE}",
                        "source": SMOKE_SOURCE,
                    })
                    found = await call("ctx_search", {
                        "queries": [marker],
                        "source": SMOKE_SOURCE,
                    })
                    if STORED_SENTENCE not in found:
                        fail(
                            "ctx_search did not retrieve the stored sentence "
                            f"(output: {found!r})"
                        )
                    print("ok: ctx_search retrieved indexed content stored under a fresh marker")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean-distribution smoke against the nested real engine.",
    )
    parser.add_argument(
        "--home", required=True,
        help="profile home passed to context-mode-hermes --install-engine",
    )
    arguments = parser.parse_args()

    check_import_origin()

    binary = (
        Path(arguments.home) / "context-mode-runtime"
        / "node_modules" / ".bin" / "context-mode"
    )
    if not binary.is_file() or not os.access(binary, os.X_OK):
        fail(f"nested engine binary missing or not executable: {binary}")
    print(f"ok: nested engine binary present at {binary}")

    asyncio.run(engine_round_trip(binary))
    print("clean-distribution smoke passed: execution, indexing and search are real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
