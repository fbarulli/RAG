#!/usr/bin/env bash
# ============================================================
# Audit — Verify Day 1 + Day 2 cleanup is complete and clean
# Run from project root.
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()    { echo -e "  ${GREEN}✔ $*${NC}"; }
warn()  { echo -e "  ${YELLOW}⚠ $*${NC}"; WARNINGS=$((WARNINGS+1)); }
fail()  { echo -e "  ${RED}✘ $*${NC}"; FAILURES=$((FAILURES+1)); }
section() { echo -e "\n${CYAN}▸ $*${NC}"; }

WARNINGS=0
FAILURES=0

[[ -f pyproject.toml ]] || { echo -e "${RED}Run from project root.${NC}"; exit 1; }

# ────────────────────────────────────────────────────────────
section "Syntax check — every .py in src/ and ablation/"
# ────────────────────────────────────────────────────────────
python3 << 'PYEOF'
import ast, sys
from pathlib import Path

errors = []
checked = 0
for p in sorted(Path(".").rglob("*.py")):
    s = str(p)
    if any(skip in s for skip in (".venv", "__pycache__", ".git")):
        continue
    checked += 1
    try:
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    except SyntaxError as e:
        errors.append(f"    {p}:{e.lineno}  {e.msg}")

print(f"  checked {checked} files")
if errors:
    for e in errors:
        print(e)
    sys.exit(len(errors))
PYEOF
[ $? -eq 0 ] && ok "no syntax errors" || fail "syntax errors found (see above)"

# ────────────────────────────────────────────────────────────
section "Stale fragments — leftover partial lines from botched removals"
# ────────────────────────────────────────────────────────────
python3 << 'PYEOF'
import re, sys
from pathlib import Path

# Patterns that should never appear as standalone lines in clean code
STALE = [
    (re.compile(r'^\s*\.resolve\(\)\.parents\[\d+\]\s*\)'),  "orphaned .resolve().parents[N])"),
    (re.compile(r'^\s*str\(Path\(__file__\)'),                "orphaned str(Path(__file__))"),
    (re.compile(r'^\s*\.parents\[\d+\]\)'),                   "orphaned .parents[N])"),
    
]

hits = []
for p in Path(".").rglob("*.py"):
    if any(s in str(p) for s in (".venv", "__pycache__", ".git")):
        continue
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        for pat, label in STALE:
            if pat.match(line):
                hits.append(f"    {p}:{i}  [{label}]  {line.strip()!r}")

if hits:
    for h in hits:
        print(h)
    sys.exit(len(hits))
print("  none found")
PYEOF
[ $? -eq 0 ] && ok "no stale fragments" || fail "stale fragments found (see above)"

# ────────────────────────────────────────────────────────────
section "Day 1 checks"
# ────────────────────────────────────────────────────────────
python3 << 'PYEOF'
import re, sys
from pathlib import Path

checks = [
    ("src.rag_pipeline imports",  re.compile(r'\bfrom src\.rag_pipeline\b|\bimport src\.rag_pipeline\b')),
    ("rag_pipeline.core.logging", re.compile(r'from rag_pipeline\.core\.logging import')),
    ("sys.path.insert hacks",     re.compile(r'\bsys\.path\.insert\s*\(')),
]

all_ok = True
for label, pat in checks:
    hits = []
    for p in Path(".").rglob("*.py"):
        if any(s in str(p) for s in (".venv", "__pycache__", ".git")):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line):
                hits.append(f"    {p}:{i}  {line.strip()!r}")
    if hits:
        print(f"  FAIL  {label} ({len(hits)} occurrence(s)):")
        for h in hits:
            print(h)
        all_ok = False
    else:
        print(f"  ok    {label}")

sys.exit(0 if all_ok else 1)
PYEOF
[ $? -eq 0 ] && ok "Day 1 patterns clean" || fail "Day 1 patterns still present"

# ────────────────────────────────────────────────────────────
section "Day 2 checks"
# ────────────────────────────────────────────────────────────
python3 << 'PYEOF'
import re, sys
from pathlib import Path

all_ok = True

# 2.1 — ablation at root should be gone
if Path("ablation").exists():
    print("  FAIL  ablation/ still exists at project root")
    all_ok = False
else:
    print("  ok    ablation/ removed from root")

# 2.1 — ablation inside rag_pipeline
if not Path("src/rag_pipeline/ablation/__init__.py").exists():
    print("  FAIL  src/rag_pipeline/ablation/__init__.py missing")
    all_ok = False
else:
    print("  ok    src/rag_pipeline/ablation/ present with __init__.py")

# 2.1 — no stale `from ablation.` imports (should all be rag_pipeline.ablation)
stale_ablation = re.compile(r'\bfrom ablation\.\b|\bimport ablation\b')
hits = []
for p in Path(".").rglob("*.py"):
    if any(s in str(p) for s in (".venv", "__pycache__", ".git")):
        continue
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if stale_ablation.search(line):
            hits.append(f"    {p}:{i}  {line.strip()!r}")
if hits:
    print(f"  FAIL  stale bare `ablation` imports ({len(hits)}):")
    for h in hits:
        print(h)
    all_ok = False
else:
    print("  ok    no stale bare ablation imports")

# 2.3 — Paths._require exists
paths_candidates = [
    Path("src/rag_pipeline/core/paths.py"),
    Path("src/rag_pipeline/paths.py"),
]
paths_py = next((p for p in paths_candidates if p.exists()), None)
if paths_py is None:
    print("  WARN  paths.py not found to check")
elif "_require" not in paths_py.read_text(encoding="utf-8"):
    print(f"  FAIL  Paths._require not found in {paths_py}")
    all_ok = False
else:
    print(f"  ok    Paths._require present in {paths_py}")

sys.exit(0 if all_ok else 1)
PYEOF
[ $? -eq 0 ] && ok "Day 2 structure clean" || fail "Day 2 issues found"

# ────────────────────────────────────────────────────────────
section "pyproject.toml sanity"
# ────────────────────────────────────────────────────────────
python3 << 'PYEOF'
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
all_ok = True

name = data.get("project", {}).get("name", "")
if name != "rag-pipeline":
    print(f"  FAIL  project name is {name!r}, expected 'rag-pipeline'")
    all_ok = False
else:
    print(f"  ok    project name: {name}")

where = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {}).get("where", [])
if "src" not in where:
    print(f"  FAIL  packages.find.where does not include 'src': {where}")
    all_ok = False
else:
    print(f"  ok    packages.find.where includes 'src'")

sys.exit(0 if all_ok else 1)
PYEOF
[ $? -eq 0 ] && ok "pyproject.toml clean" || fail "pyproject.toml issues found"

# ────────────────────────────────────────────────────────────
section "Import smoke tests"
# ────────────────────────────────────────────────────────────

smoke() {
  local label="$1"; local code="$2"
  if uv run python -c "$code" > /dev/null 2>&1; then
    ok "$label"
  else
    fail "$label"
    uv run python -c "$code" 2>&1 | sed 's/^/    /'
    FAILURES=$((FAILURES+1))
  fi
}

smoke "rag_pipeline.logging"            "from rag_pipeline.logging import get_logger"
smoke "rag_pipeline.ablation"           "import rag_pipeline.ablation"
smoke "rag_pipeline.ablation.cli"       "from rag_pipeline.ablation.cli import main"
smoke "rag_pipeline.eda.topics.config"  "from rag_pipeline.eda.topics.config import TopicsConfig"
smoke "rag_pipeline.core.paths"         "from rag_pipeline.core.paths import Paths"
smoke "Paths.entity_patterns() exists"  "from rag_pipeline.core.paths import Paths; assert Paths.entity_patterns().exists()"

# ────────────────────────────────────────────────────────────
section "configs/ sanity"
# ────────────────────────────────────────────────────────────
python3 << 'PYEOF'
import sys
from pathlib import Path

expected = [
    "configs/entity_patterns.json",
    "configs/paths.json",
]
all_ok = True
for f in expected:
    p = Path(f)
    if p.exists():
        print(f"  ok    {f}")
    else:
        print(f"  FAIL  {f} missing")
        all_ok = False

# entity_patterns should NOT still be in the old location
old = Path("src/rag_pipeline/eda/topics/entity_patterns.json")
if old.exists():
    print(f"  FAIL  {old} still exists (should have been moved to configs/)")
    all_ok = False
else:
    print(f"  ok    entity_patterns not in old src/ location")

sys.exit(0 if all_ok else 1)
PYEOF
[ $? -eq 0 ] && ok "configs clean" || fail "configs issues found"

# ────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [[ $FAILURES -eq 0 && $WARNINGS -eq 0 ]]; then
  echo -e "${GREEN}  All checks passed — Day 1 + Day 2 are clean ✓${NC}"
elif [[ $FAILURES -eq 0 ]]; then
  echo -e "${YELLOW}  $WARNINGS warning(s), 0 failures — review above${NC}"
else
  echo -e "${RED}  $FAILURES failure(s), $WARNINGS warning(s) — fix before Day 3${NC}"
fi
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"