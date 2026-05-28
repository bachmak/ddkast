#!/usr/bin/env python3

import json
import os
import subprocess
import sys

data = json.load(sys.stdin)
fp = data.get("tool_input", {}).get("file_path", "")

if fp.endswith(".py") and os.path.exists(fp):
    subprocess.run(["uv", "run", "ruff", "format", fp])
    subprocess.run(["uv", "run", "ruff", "check", "--fix", fp])
