"""Rebuild ``results/SUMMARY.md`` from every evaluation run on disk.

Each run of demo 16 leaves ``results/runs/<name>.json`` and never overwrites one,
so the folder is the complete record. This turns it into one table, newest last.
The summary itself is a view: delete it and run this again and it comes back.

Run from the repository root:

    python .\\demos\\06_imitation_learning\\tools\\23_summarize_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def short(text: str, limit: int = 70) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> int:
    runs = []
    for path in sorted((RESULTS / "runs").glob("*.json")):
        if path.name.startswith("smoke"):
            continue  # script smoke tests, git-ignored; not results
        record = json.loads(path.read_text(encoding="utf-8"))
        runs.append((record["run"]["utc"], path.name, record))
    runs.sort()

    lines = [
        "# Resumen de evaluaciones",
        "",
        "Generado por `tools/23_summarize_results.py` a partir de `results/runs/*.json`; "
        "no se edita a mano. Cada ejecución del demo 16 deja su propio archivo y nunca "
        "sobrescribe otro.",
        "",
        "| fecha (UTC) | ejecución | tipo | entradas | cubos | éxitos | fallos |",
        "|---|---|---|---|---|---|---|",
    ]
    for utc, name, record in runs:
        run, summary = record["run"], record["summary"]
        failures = ", ".join(f"{k}×{v}" for k, v in sorted(summary["failures"].items())) or "-"
        lines.append(
            f"| {utc} | [`{name[:-5]}`](runs/{name[:-5]}.md) | {run['type']} | {short(run['inputs'])} "
            f"| {run['spots']} | **{summary['successes']}/{summary['episodes']}** | {failures} |"
        )
    if not runs:
        lines.append("| (ninguna todavía) | | | | | | |")

    offline = sorted((RESULTS / "offline").glob("*.json")) if (RESULTS / "offline").is_dir() else []
    if offline:
        lines += ["", "## Comprobaciones sin simulador (`tools/22_validate_offline.py`)", "",
                  "| ejecución | política | episodios | error xy al cerrar la pinza (mm) | error z (mm) |",
                  "|---|---|---|---|---|"]
        for path in offline:
            record = json.loads(path.read_text(encoding="utf-8"))
            parts = Path(record["policy"]).parts
            who = parts[-4] if len(parts) > 4 else record["policy"]
            xy, z = record["grasp_xy_error_mm"], record["grasp_z_error_mm"]
            lines.append(
                f"| `{path.stem}` | `{who}` | {record['episodes']} "
                f"| media {xy['mean']:.1f}, peor {xy['worst']:.1f} "
                f"| media {z['mean']:.1f}, peor {z['worst']:.1f} |"
            )

    (RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(runs)} run(s) summarised in {RESULTS / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
