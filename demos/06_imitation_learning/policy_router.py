"""Resolve classifier outcomes to reusable post-classification policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("policy_routes.json")


def upload_time(metadata: dict) -> datetime:
    value = metadata.get("uploaded_at")
    return datetime.fromisoformat(value) if value else datetime.min.replace(tzinfo=timezone.utc)


def upload_label(metadata: dict) -> str:
    if not metadata.get("uploaded_at"):
        return "fecha de subida no disponible"
    return upload_time(metadata).astimezone(ZoneInfo("Europe/Madrid")).strftime("%d/%m/%Y %H:%M:%S %Z")


@dataclass(frozen=True)
class PolicyRoute:
    name: str
    runner: Path
    checkpoint: Path
    max_seconds: float
    n_action_steps: int
    max_relative_target: float
    raise_clearance: bool
    hub_repo: str | None = None
    hub_path: str | None = None
    hub_revision: str | None = None
    dataset_features: Path | None = None


class PolicyRouteRegistry:
    """Map a classifier label to a policy runner and checkpoint."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self._classifier_routes = config["classifier_routes"]
        self._destinations = config.get("destinations", {})
        self._checkpoint_options = config.get("checkpoint_options", {})
        self._checkpoint_hubs = config.get("checkpoint_hubs", {})
        self._pick_checkpoint_author = config.get("pick_checkpoint_author")
        self._pick_checkpoints = config.get("pick_checkpoints", {})
        self._policies = config["policies"]
        if not self._classifier_routes or not self._policies:
            raise ValueError("Policy route configuration needs classifier routes and policies")
        for label, name in self._classifier_routes.items():
            if name not in self._policies:
                raise ValueError(f"Classifier label {label!r} refers to unknown policy {name!r}")
        for destination, name in self._destinations.items():
            if name not in self._policies:
                raise ValueError(f"Destination {destination!r} refers to unknown policy {name!r}")
        for policy, options in self._checkpoint_options.items():
            if policy not in self._policies or any(name not in self._policies for name in options.values()):
                raise ValueError(f"Checkpoint options for {policy!r} refer to an unknown policy")

    def refresh_checkpoints(self) -> None:
        """Discover uploaded weights at one Hub revision before changing the active catalog."""
        from huggingface_hub import HfApi

        api = HfApi()
        policies, catalogs = {}, {}
        for family, repo in self._checkpoint_hubs.items():
            revision = api.model_info(repo).sha
            files = list(api.list_repo_tree(repo, recursive=True, expand=True, revision=revision))
            known = {policy.get("hub_path"): name for name, policy in self._policies.items()
                     if policy.get("hub_repo") == repo}
            options = {}
            for file in files:
                if not file.path.endswith("/pretrained_model/model.safetensors"):
                    continue
                source = PurePosixPath(file.path).parent
                step = source.parent.name
                if not step.isdigit():
                    continue
                run = source.parent.parent.parent.name
                name = known.get(str(source), f"{family}-{run}-{step}")
                policy = dict(self._policies.get(name, self._policies[family]))
                if name not in self._policies:
                    policy["checkpoint"] = str(Path(self._policies[family]["checkpoint"]).parents[1]
                                               / "checkpoints" / run / step / "pretrained_model")
                policy.update(hub_repo=repo, hub_path=str(source), hub_revision=revision,
                              uploaded_at=file.last_commit.date.isoformat())
                policies[name] = policy
                step_label = f"{int(step):,}".replace(",", ".")
                # Keep different training runs selectable even when they share a step number.
                label = step_label if step_label not in options else f"{step_label} ({run})"
                options[label] = name
            if not options:
                raise ValueError(f"No ACT checkpoints found in {repo}")
            catalogs[family] = options
        self._policies.update(policies)
        self._checkpoint_options.update(catalogs)

    def refresh_pick_checkpoints(self) -> None:
        """Use the upload time of the weights, rather than README edits, for pick models."""
        if not self._pick_checkpoint_author:
            return
        from huggingface_hub import HfApi

        api = HfApi()
        checkpoints = {}
        for model in api.list_models(author=self._pick_checkpoint_author, search="act-pick", full=True):
            if not model.id.split("/")[-1].startswith("act-pick-"):
                continue
            for file in api.list_repo_tree(model.id, recursive=True, expand=True, revision=model.sha):
                if file.path == "model.safetensors":
                    checkpoints[model.id] = {"revision": model.sha,
                                             "uploaded_at": file.last_commit.date.isoformat()}
        if checkpoints:
            self._pick_checkpoints = checkpoints

    def pick_checkpoint_options(self, current_policy: str | None = None) -> dict[str, str]:
        choices = {
            f"{repo.split('/')[-1]} · {upload_label(metadata)}": repo
            for repo, metadata in sorted(self._pick_checkpoints.items(),
                                         key=lambda item: upload_time(item[1]), reverse=True)
        }
        if current_policy and current_policy not in choices.values():
            choices[f"{current_policy} · fecha de subida no disponible"] = current_policy
        return choices

    def pick_revision(self, repo: str) -> str | None:
        return self._pick_checkpoints.get(repo, {}).get("revision")

    def resolve(self, classifier_label: str) -> PolicyRoute:
        try:
            name = self._classifier_routes[classifier_label]
            policy = self._policies[name]
        except KeyError as exc:
            raise ValueError(f"No post-classification policy configured for {classifier_label!r}") from exc
        return self._route(name)

    def checkpoint_options(self, destination: str) -> tuple[str, ...]:
        try:
            policy = self._destinations[destination]
        except KeyError as exc:
            raise ValueError(f"No post-classification policy configured for destination {destination!r}") from exc
        return tuple(self._checkpoint_choices(policy))

    def _checkpoint_choices(self, policy: str) -> dict[str, str]:
        options = self._checkpoint_options.get(policy, {})

        choices = {}
        for label, name in sorted(options.items(), key=lambda item: upload_time(self._policies[item[1]]),
                                  reverse=True):
            date = upload_label(self._policies[name])
            choices[f"{label} pasos · {date}"] = name
        return choices

    def resolve_destination(self, destination: str, checkpoint: str | None = None) -> PolicyRoute:
        try:
            name = self._destinations[destination]
        except KeyError as exc:
            raise ValueError(f"No post-classification policy configured for destination {destination!r}") from exc
        options = {**self._checkpoint_options.get(name, {}), **self._checkpoint_choices(name)}
        if checkpoint is not None and options:
            try:
                name = options[checkpoint]
            except KeyError as exc:
                raise ValueError(f"Unknown checkpoint {checkpoint!r} for destination {destination!r}") from exc
        return self._route(name)

    def _route(self, name: str) -> PolicyRoute:
        policy = self._policies[name]
        def repo_path(key: str) -> Path:
            path = (REPO_ROOT / policy[key]).resolve()
            if not path.is_relative_to(REPO_ROOT):
                raise ValueError(f"Policy {name!r} {key} must stay inside the repository")
            return path

        max_seconds = float(policy["max_seconds"])
        n_action_steps = int(policy["n_action_steps"])
        max_relative_target = float(policy["max_relative_target"])
        if max_seconds <= 0 or n_action_steps <= 0 or max_relative_target <= 0:
            raise ValueError(f"Policy {name!r} limits must be positive")
        return PolicyRoute(name, repo_path("runner"), repo_path("checkpoint"), max_seconds,
                           n_action_steps, max_relative_target, bool(policy.get("raise_clearance", False)),
                           policy.get("hub_repo"), policy.get("hub_path"), policy.get("hub_revision"),
                           repo_path("dataset_features") if policy.get("dataset_features") else None)
