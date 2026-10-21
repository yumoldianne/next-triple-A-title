"""Run the whole pipeline (works on Windows, macOS, Linux):

    python run_all.py        run every step
    python run_all.py 04     resume from step 04 (earlier outputs are kept)

Everything printed is also saved to outputs/run_log.txt (appended when resuming)."""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["01_build_warehouse.py", "02_eda.py", "03_hit_metric.py",
         "04_features.py", "05_model.py", "06_report.py"]

# Windows consoles default to cp1252, which can't print characters like ✓ or —.
# Force every step (and this runner) to speak UTF-8.
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

START = sys.argv[1].zfill(2) if len(sys.argv) > 1 else "01"
(ROOT / "outputs").mkdir(exist_ok=True)
log = open(ROOT / "outputs/run_log.txt", "w" if START == "01" else "a", encoding="utf-8")
for step in [s for s in STEPS if s[:2] >= START]:
    t0 = time.time()
    print(f"\n=== {step} ===", flush=True); log.write(f"\n=== {step} ===\n")
    p = subprocess.Popen([sys.executable, "-u", step], cwd=ROOT / "src", env=ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        if "Warning" in line:
            continue
        print(line, end=""); log.write(line); log.flush()
    if p.wait() != 0:
        sys.exit(f"\n{step} failed - see outputs/run_log.txt")
    print(f"--- {step} done in {time.time() - t0:.0f}s"); log.write(f"--- done {time.time()-t0:.0f}s\n")
print("\nAll done.")