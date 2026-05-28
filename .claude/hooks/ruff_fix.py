#!/usr/bin/env python3

import json
import subprocess
import sys

data = json.load(sys.stdin)
fp = data.get("tool_input", {}).get("file_path", "")

if fp.endswith(".py"):
    subprocess.run(["uv", "run", "ruff", "format", fp], capture_output=True)
