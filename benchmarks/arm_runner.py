"""Run ONE (arm, workload) against the real pinned Hermes AIAgent, in-process.

Invoked as a module by run_bench.py so every arm gets a pristine process:
its own HERMES_HOME profile, its own plugin manager, its own MCP servers.
Nothing is monkeypatched — plugins, hooks, MCP discovery, the native context
compressor and the terminal tool all run exactly as a native Hermes session
would. Cleanup is explicit: agent.close(), MCP shutdown, then os._exit so no
Hermes background thread can outlive the benchmark child.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _fail(out: Path, error: str) -> int:
    out.write_text(json.dumps({"ok": False, "error": error}, indent=2), encoding="utf-8")
    print(error, file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("small", "large"), required=True)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--nonce")
    parser.add_argument("--fixture-url")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--arm", choices=("baseline", "plugin"), required=True)
    args = parser.parse_args()

    home = args.home.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(home)
    os.chdir(workdir)

    # Bind Hermes's context-local home BEFORE importing agent modules so every
    # config/plugin/MCP read resolves against this arm's profile only.
    import hermes_constants
    token = hermes_constants.set_hermes_home_override(home)

    result: dict = {"arm": args.arm, "workload": args.workload, "ok": False}
    agent = None
    try:
        # Native, config-driven MCP discovery (no-op when the profile has no
        # mcp_servers, i.e. the baseline arm).
        from tools.mcp_tool_discovery import discover_mcp_tools
        try:
            discover_mcp_tools()
            result["mcp_discovery"] = "ok"
        except Exception as exc:
            result["mcp_discovery"] = f"failed: {type(exc).__name__}: {exc}"[:500]

        from run_agent import AIAgent

        toolsets = ["terminal"] + (["mcp-context-mode"] if args.arm == "plugin" else [])
        queries = {
            "small": (
                f"Run exactly this shell command with the terminal tool, then report its "
                f"exact stdout and nothing else: printf '{args.nonce}'"
            ),
            "large": (
                f"Fetch {args.fixture_url} with the terminal tool using curl -sS, then answer "
                "from the actual data: the JSON has a 'records' array; report COUNT = number "
                "of records and SUM = sum of every record's 'value' field, as "
                "'COUNT=<n> SUM=<s>'."
            ),
        }
        started = time.monotonic()
        toolsets = ["terminal"] + ([] if args.arm == "baseline" else ["mcp-context-mode"])
        agent = AIAgent(
            model=args.model,
            provider="custom",
            base_url=args.base_url,
            api_key=args.api_key,
            max_iterations=args.max_iterations,
            enabled_toolsets=toolsets,
            quiet_mode=True,
            session_id=args.session_id,
            skip_context_files=True,
            skip_memory=True,
            skip_background_review=True,
            save_trajectories=False,
            cwd=workdir,
            platform="bench",
        )
        try:
            result["agent_tools"] = sorted({t.get("function", {}).get("name") or t.get("name") or "?"
                                            for t in (agent.tools or [])})
        except Exception:
            result["agent_tools"] = []
        result["n_agent_tools"] = len(result["agent_tools"])
        result["ctx_tools_advertised"] = [n for n in result["agent_tools"] if "ctx_" in n]
        try:
            result["effective_context_length"] = int(getattr(agent, "context_length", 0) or 0)
        except Exception:
            result["effective_context_length"] = None

        conversation = agent.run_conversation(queries[args.workload])
        result["wall_s"] = round(time.monotonic() - started, 3)

        messages = conversation.get("messages") or []
        final_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_text = msg["content"] if isinstance(msg["content"], str) else json.dumps(msg["content"])
                break
        tool_calls = []
        for msg in messages:
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                tool_calls.append({"name": fn.get("name"),
                                   "arguments_head": str(fn.get("arguments"))[:400]})
        tool_results = [{"chars": len(m.get("content") or "")} for m in messages if m.get("role") == "tool"]
        result.update({
            "ok": True,
            "completed": bool(conversation.get("completed")),
            "final_text": final_text[:2000],
            "n_messages": len(messages),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        })
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        result["traceback"] = traceback.format_exc()[-4000:]
    finally:
        for cleanup in ("agent.close()", "shutdown_mcp_servers()"):
            try:
                if cleanup == "agent.close()":
                    if agent is not None:
                        agent.close()
                else:
                    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
                    shutdown_mcp_servers()
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(f"{cleanup}: {type(exc).__name__}: {exc}"[:300])
        try:
            hermes_constants.reset_hermes_home_override(token)
        except Exception:
            pass
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    raise SystemExit(main())
