#!/usr/bin/env python3
"""
Skill execution harness — invoked as a subprocess by executor.py.

Security properties:
  - Runs in a caller-supplied temp working directory (exec_dir passed as cwd by Popen)
  - Parent passes env={PATH: ...} only — no secrets accessible
  - Stdout carries the JSON result; stderr is captured but not trusted
  - Hard 30s wall-clock limit enforced by the parent's wait_for() timeout

Usage:  python run_harness.py <skill_path.py>
Stdin:  JSON dict   (input_data for run())
Stdout: JSON object {"ok": true, "result": {...}}
                 or {"ok": false, "error": "..."}
"""
import sys
import json
import traceback
import importlib.util


def main() -> None:
    if len(sys.argv) < 2:
        _fail("no skill path given")
        return

    skill_path = sys.argv[1]

    try:
        raw_stdin = sys.stdin.read()
        input_data = json.loads(raw_stdin) if raw_stdin.strip() else {}
    except json.JSONDecodeError as exc:
        _fail(f"invalid input JSON: {exc}")
        return

    try:
        spec = importlib.util.spec_from_file_location("_generated_skill", skill_path)
        if spec is None or spec.loader is None:
            _fail(f"could not load skill from {skill_path}")
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        _fail(f"skill load error: {exc}\n{traceback.format_exc()[-1000:]}")
        return

    if not hasattr(mod, "run"):
        _fail("skill has no run() function")
        return

    try:
        result = mod.run(input_data)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}")
        return

    if not isinstance(result, dict):
        result = {"result": result}

    print(json.dumps({"ok": True, "result": result}))


def _fail(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg[:2000]}))
    sys.exit(1)


if __name__ == "__main__":
    main()
