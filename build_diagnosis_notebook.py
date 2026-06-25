"""
Convert regime_diagnosis_nb.py → regime_diagnosis.ipynb
"""
import re, json
from pathlib import Path
from uuid import uuid4

SRC  = Path("regime_diagnosis_nb.py")
DEST = Path("regime_diagnosis.ipynb")

text  = SRC.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)

cells_raw, current_kind, current_label, current_body = [], None, "", []
CELL_RE = re.compile(r"^# %%(?P<md> \[markdown\])?(?P<label>.*)")


def flush(kind, label, body):
    if kind is not None and any(l.strip() for l in body):
        cells_raw.append((kind, label.strip(), body[:]))


for line in lines:
    m = CELL_RE.match(line)
    if m:
        flush(current_kind, current_label, current_body)
        current_kind  = "markdown" if m.group("md") else "code"
        current_label = m.group("label") or ""
        current_body  = []
    else:
        current_body.append(line)

flush(current_kind, current_label, current_body)


def make_cell(cell_type, source, cell_id):
    base = {"cell_type": cell_type, "id": cell_id, "metadata": {}, "source": source}
    if cell_type == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


nb_cells = []
for kind, label, body in cells_raw:
    cell_id = uuid4().hex[:8]
    if kind == "markdown":
        md_lines = []
        for l in body:
            s = l.rstrip("\n")
            md_lines.append((s[2:] if s.startswith("# ") else ("" if s == "#" else s)) + "\n")
        source = "".join(md_lines).strip()
        if source:
            nb_cells.append(make_cell("markdown", source, cell_id))
    else:
        source = "".join(body).rstrip("\n")
        if source.strip():
            nb_cells.append(make_cell("code", source, cell_id))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"},
    },
    "cells": nb_cells,
}

DEST.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {len(nb_cells)} cells → {DEST}")
print(f"  code     : {sum(1 for c in nb_cells if c['cell_type'] == 'code')}")
print(f"  markdown : {sum(1 for c in nb_cells if c['cell_type'] == 'markdown')}")
