from __future__ import annotations

import sys
import time
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for package_name, package_path in (
    ("backend", PROJECT_ROOT / "backend"),
    ("backend.services", PROJECT_ROOT / "backend" / "services"),
):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from backend.services.own_llm import own_generate


PROMPT = """Summarize changes between two lab reports (earlier -> later) in under 150 words. Use ONLY this data; say improved/worsened/stable; no diagnosis.
[{"name": "HbA1c", "unit": "%", "old": 7.4, "new": 6.1, "delta": -1.3, "old_flag": "high", "new_flag": "high"}, {"name": "LDL Cholesterol", "unit": "mg/dL", "old": 96.0, "new": 128.5, "delta": 32.5, "old_flag": "normal", "new_flag": "high"}]"""


def main() -> int:
    started = time.perf_counter()
    answer = own_generate(PROMPT, 200)
    elapsed = time.perf_counter() - started
    print(answer)
    print(f"Seconds taken: {elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
