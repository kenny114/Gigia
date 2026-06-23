#!/usr/bin/env python3
"""
Skill execution harness — run as a subprocess by executor.py.

Usage:  python run_harness.py <skill_path.py>
Input:  JSON dict on stdin  (the input_data argument)
Output: JSON on stdout      {"ok": true, "result": {...}}
                        or  {"ok": false, "error": "..."}

This script runs with no parent env vars (executor passes env={}),
so secrets from the Gigia container cannot be exfiltrated.
"""
import sys
import json
import traceback
import importlib.util


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "no skill path given"}))
        sys.exit(1)

    skill_path = sys.argv[1]

    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        print(json.dumps({"ok": False, "error": f"invalid input JSON: {e}"}))
        sys.exit(1)

    try:
        spec = importlib.util.spec_from_file_location("_generated_skill", skill_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"skill load error: {e}"}))
        sys.exit(1)

    if not hasattr(mod, "run"):
        print(json.dumps({"ok": False, "error": "skill has no run() function"}))
        sys.exit(1)

    try:
        result = mod.run(input_data)
        if not isinstance(result, dict):
            result = {"result": result}
        print(json.dumps({"ok": True, "result": result}))
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()[-2000:],
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
