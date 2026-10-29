"""
Run the code of the spatial networks concept note and write each block's output into the page.

    python docs/scripts/run_concept_examples.py [--check] [--workdir DIR]

The pages of ``docs/concepts/spatial-networks/`` are read in the order of
``PAGES``. Their ``python`` blocks run one after another in one namespace, as
a reader working through the note would run them. When a ``python`` block is
followed by a ``text`` block, that block is replaced by what the code printed.
Blocks that should not run start with ``# not run``.

The code runs in ``--workdir`` (``/tmp/gs_concepts`` by default), so the
handler's downloads and saved networks stay out of the repository. It needs
the internet the first time, for Rwanda's extract and for Overpass.

``--check`` runs the code without writing, and fails if an output changed in
a way other than timings.
"""

import argparse
import contextlib
import io
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES_DIR = ROOT / "docs" / "concepts" / "spatial-networks"
PAGES = [
    "index.md",
    "getting-a-network.md",
    "data-model.md",
    "editing.md",
    "inserting-points.md",
    "reduction.md",
    "algorithms.md",
    "storage.md",
    "use-cases.md",
    "performance.md",
]
BLOCK = re.compile(r"```python\n(.*?)```(\n\n```text\n.*?```)?", re.DOTALL)
TIMING = re.compile(r"\d+(\.\d+)? (s|ms)\b")


def run_page(text: str, namespace: dict, page: str) -> str:
    pieces, last = [], 0
    for match in BLOCK.finditer(text):
        code, output_block = match.group(1), match.group(2)
        pieces.append(text[last : match.start()])
        last = match.end()
        if code.lstrip().startswith("# not run"):
            pieces.append(match.group(0))
            continue
        buffer = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, f"{page}", "exec"), namespace)
        print(f"  {page}: block ran in {time.perf_counter() - started:.1f} s", file=sys.stderr)
        printed = buffer.getvalue().rstrip("\n")
        if output_block is None:
            pieces.append(match.group(0))
        else:
            pieces.append(f"```python\n{code}```\n\n```text\n{printed}\n```")
    pieces.append(text[last:])
    return "".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--workdir", default="/tmp/gs_concepts")
    args = parser.parse_args()
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    namespace: dict = {"__name__": "__concepts__"}
    changed = []
    for page in PAGES:
        path = PAGES_DIR / page
        print(page, file=sys.stderr)
        before = path.read_text()
        after = run_page(before, namespace, page)
        if after != before:
            if TIMING.sub("T", after) != TIMING.sub("T", before):
                changed.append(page)
            if not args.check:
                path.write_text(after)
    if args.check and changed:
        raise SystemExit(f"Outputs changed in: {', '.join(changed)}")


if __name__ == "__main__":
    main()
