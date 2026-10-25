"""Refresh the pick-policy catalog in ``policy_routes.json`` from the Hugging Face Hub.

Lists every ``IkerAzurmendi/act-pick-*`` repo (one checkpoint each) and records, per repo, the revision, the
upload time of the weights and what its model card says about it: training episodes, chunk size and the offline
error on the held-out episodes (mean absolute joint error, and how far below the "stay still" baseline it is).
The apps show these facts next to each checkpoint, and use the catalog as is when the Hub cannot be reached.

The offline error is a check on 15-16 held-out episodes with one seed; it does not say whether the arm grasps.
Errors of different chunk sizes are not comparable, the "below stay-still" percentage is.

    python .\\demos\\06_imitation_learning\\analyze_pick_policies.py
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROUTES = Path(__file__).with_name("policy_routes.json")
ROW_NUMBERS = re.compile(r"\*{0,2}(\d+\.\d+)\*{0,2}\s*\|\s*\*{0,2}(\d+)%")


def card_facts(readme: str, repo: str) -> dict:
    """Error and percentage below stay-still of this very checkpoint, if its card gives them."""
    name = repo.split("/")[-1]
    rows = [line for line in readme.splitlines() if line.startswith("|")]
    mine = [r for r in rows if "this one" in r or "this model" in r] or [r for r in rows if r.rstrip(" |").endswith(name)]
    found = ROW_NUMBERS.search(mine[0]) if mine else None
    facts = {"error": float(found[1]), "below": int(found[2])} if found else {}
    episodes = re.search(r"(\d+) training episodes", readme)
    return {**({"episodes": int(episodes[1])} if episodes else {}), **facts}


def main() -> int:
    try:  # this PC's Python does not trust the Windows certificate store by itself (see 19_build_so101_act_dataset.py)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from huggingface_hub import HfApi, hf_hub_download

    config = json.loads(ROUTES.read_text(encoding="utf-8"))
    author, api = config["pick_checkpoint_author"], HfApi()

    def fetch(model):
        weights = next(f for f in api.list_repo_tree(model.id, revision=model.sha, expand=True)
                       if f.path == "model.safetensors")
        chunk = json.loads(Path(hf_hub_download(model.id, "config.json", revision=model.sha)).read_text())["chunk_size"]
        try:
            readme = Path(hf_hub_download(model.id, "README.md", revision=model.sha)).read_text(encoding="utf-8")
        except Exception:  # the first round (act-pick-N) has no model card
            readme = ""
        return model.id, {"revision": model.sha, "uploaded_at": weights.last_commit.date.isoformat(), "chunk": chunk,
                          **card_facts(readme, model.id)}

    models = [m for m in api.list_models(author=author, search="act-pick", full=True)
              if m.id.split("/")[-1].startswith("act-pick-")]
    with ThreadPoolExecutor(8) as pool:
        config["pick_checkpoints"] = dict(sorted(pool.map(fetch, models)))
    ROUTES.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    missing = [r for r, v in config["pick_checkpoints"].items() if "error" not in v]
    print(f"{len(config['pick_checkpoints'])} checkpoints written to {ROUTES.name}; no offline error on the card of: "
          f"{', '.join(r.split('/')[-1] for r in missing) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
