"""Run auth tests and print clean output (filters structlog noise)."""
import subprocess
import re

r = subprocess.run(
    [r".\.venv\Scripts\python", "-m", "unittest", "tests.test_auth", "-v"],
    capture_output=True,
    text=True,
)
for line in r.stderr.replace("\r", "").split("\n"):
    s = line.strip()
    if not s:
        continue
    if s.startswith("{") and '"event"' in s:
        continue
    if "HTTP Request" in s:
        continue
    print(s)

print()
print("EXIT CODE:", r.returncode)
