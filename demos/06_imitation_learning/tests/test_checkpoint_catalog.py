"""Hub upload dates and UI selections must resolve to the matching ACT snapshot."""

import importlib.util
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch


DEMO_DIR = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


router = load_module("checkpoint_catalog_router", DEMO_DIR / "policy_router.py")
app = load_module("checkpoint_catalog_app", DEMO_DIR / "24_pick_and_classify_app.py")


def hub_file(step, uploaded, run="act_bad_v1"):
    return SimpleNamespace(
        path=f"{run}/checkpoints/{step:06d}/pretrained_model/model.safetensors",
        last_commit=SimpleNamespace(date=datetime.fromisoformat(uploaded)),
    )


class CheckpointCatalogTests(unittest.TestCase):
    def test_local_catalog_has_all_four_uploads_newest_first_in_madrid_time(self):
        registry = router.PolicyRouteRegistry()
        choices = registry.checkpoint_options("white")

        self.assertEqual(len(choices), 4)
        self.assertTrue(choices[0].startswith("25.000 pasos"))
        self.assertIn("24/10/2026 21:56:32 CEST", choices[0])
        self.assertTrue(choices[1].startswith("20.000 pasos"))
        self.assertTrue(choices[2].startswith("15.000 pasos"))
        self.assertTrue(choices[3].startswith("10.000 pasos"))
        route = registry.resolve_destination("white", choices[0])
        self.assertEqual(route.hub_path, "act_bad_v1/checkpoints/025000/pretrained_model")
        self.assertEqual(registry.resolve_destination("green", choices[0]).name, "up-down-good")

    def test_pick_catalog_lists_every_run_and_checkpoint_grouped_by_run_and_step(self):
        registry = router.PolicyRouteRegistry()
        choices = registry.pick_checkpoint_options()
        repos = list(choices.values())
        self.assertEqual(len(choices), 40)
        self.assertEqual(len(set(repos)), 40)
        self.assertEqual(repos[0], "IkerAzurmendi/act-pick-v3-6000")
        self.assertEqual(repos[-1], "IkerAzurmendi/act-pick-30000")
        runs = [router.pick_run_and_step(repo)[0] for repo in repos]
        self.assertEqual([run for i, run in enumerate(runs) if i == 0 or run != runs[i - 1]],
                         ["v3 super", "v3 strict", "v3 1-2-pieces", "v3 1piece", "v3 chunk60", "v3 recent", "v2", "v1"])
        for run in set(runs):
            steps = [router.pick_run_and_step(r)[1] for r in repos if router.pick_run_and_step(r)[0] == run]
            self.assertEqual(steps, sorted(steps), run)

    def test_pick_labels_say_run_steps_episodes_offline_error_and_madrid_upload_time(self):
        choices = router.PolicyRouteRegistry().pick_checkpoint_options()
        by_repo = {repo: label for label, repo in choices.items()}
        self.assertTrue(by_repo["IkerAzurmendi/act-pick-v3-6000"].startswith(
            "v3 super · 6.000 pasos · 398 ep · −27 % bajo reposo (error 15,16) · "))
        self.assertIn("24/10/2026 21:58:48 CEST", by_repo["IkerAzurmendi/act-pick-v2-12000"])
        # the first round has no model card: nothing is invented for it
        self.assertEqual(by_repo["IkerAzurmendi/act-pick-6000"],
                         "v1 · 6.000 pasos · 24/10/2026 15:06:57 CEST")

    def test_unknown_pick_repo_name_is_listed_with_its_own_name_after_the_known_runs(self):
        registry = router.PolicyRouteRegistry()
        registry._pick_checkpoints["IkerAzurmendi/act-pick-v4-new-100"] = {"uploaded_at": "2026-10-26T10:00:00+00:00"}
        choices = registry.pick_checkpoint_options()
        self.assertEqual(list(choices.values())[-1], "IkerAzurmendi/act-pick-v4-new-100")
        self.assertTrue(list(choices)[-1].startswith("act-pick-v4-new-100 · 26/10/2026 11:00:00 CET"))

    def test_pick_catalog_retains_a_cli_local_model_without_fabricating_a_date(self):
        registry = router.PolicyRouteRegistry()
        choices = registry.pick_checkpoint_options("/tmp/my-pick-model")
        self.assertEqual(list(choices.values())[-1], "/tmp/my-pick-model")
        self.assertIn("fecha de subida no disponible", list(choices)[-1])

    def test_pick_upload_date_comes_from_weights_not_readme(self):
        api = Mock()
        api.list_models.return_value = [SimpleNamespace(id="IkerAzurmendi/act-pick-v2-18000", sha="revision-v2")]
        api.list_repo_tree.return_value = [
            SimpleNamespace(path="README.md", last_commit=SimpleNamespace(
                date=datetime.fromisoformat("2026-10-26T18:00:00+00:00"))),
            SimpleNamespace(path="model.safetensors", last_commit=SimpleNamespace(
                date=datetime.fromisoformat("2026-10-24T18:00:00+00:00"))),
        ]
        registry = router.PolicyRouteRegistry()
        with patch("huggingface_hub.HfApi", return_value=api):
            registry.refresh_pick_checkpoints()
        choices = registry.pick_checkpoint_options()
        self.assertEqual(len(choices), 1)
        self.assertIn("24/10/2026 20:00:00 CEST", next(iter(choices)))
        # the Hub listing gives dates only: what the catalog knew about the run is kept
        self.assertIn("v2 · 18.000 pasos · 278 ep · −26 % bajo reposo", next(iter(choices)))
        self.assertEqual(registry.pick_revision("IkerAzurmendi/act-pick-v2-18000"), "revision-v2")

    def test_upload_date_orders_catalog_even_when_the_lower_step_is_newer(self):
        api = Mock()
        api.model_info.return_value.sha = "pinned-revision"
        api.list_repo_tree.return_value = [
            hub_file(25000, "2026-10-24T20:00:00+00:00"),
            hub_file(10000, "2026-10-25T08:00:00+00:00"),
            SimpleNamespace(path="act_bad_v1/checkpoints/025000/training_state/optimizer_state.safetensors"),
        ]
        registry = router.PolicyRouteRegistry()
        with patch("huggingface_hub.HfApi", return_value=api):
            registry.refresh_checkpoints()

        choices = registry.checkpoint_options("white")
        self.assertEqual(len(choices), 2)
        self.assertTrue(choices[0].startswith("10.000 pasos"))
        self.assertTrue(choices[1].startswith("25.000 pasos"))
        route = registry.resolve_destination("white", choices[1])
        self.assertEqual(route.hub_revision, "pinned-revision")
        self.assertEqual(route.hub_path, "act_bad_v1/checkpoints/025000/pretrained_model")
        api.list_repo_tree.assert_called_once_with(
            "MikelRubi/act-up-down-bad", recursive=True, expand=True, revision="pinned-revision")

    def test_same_step_in_different_runs_keeps_both_selectable(self):
        api = Mock()
        api.model_info.return_value.sha = "pinned-revision"
        api.list_repo_tree.return_value = [
            hub_file(10000, "2026-10-24T19:00:00+00:00"),
            hub_file(10000, "2026-10-25T19:00:00+00:00", run="act_bad_v2"),
        ]
        registry = router.PolicyRouteRegistry()
        with patch("huggingface_hub.HfApi", return_value=api):
            registry.refresh_checkpoints()

        choices = registry.checkpoint_options("white")
        self.assertEqual(len(choices), 2)
        self.assertIn("act_bad_v2", registry.resolve_destination("white", choices[0]).hub_path)
        self.assertIn("act_bad_v1", registry.resolve_destination("white", choices[1]).hub_path)

    def test_failed_refresh_preserves_the_local_catalog(self):
        registry = router.PolicyRouteRegistry()
        before = registry.checkpoint_options("white")
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.list_repo_tree.side_effect = OSError("offline")
            with self.assertRaisesRegex(OSError, "offline"):
                registry.refresh_checkpoints()
        self.assertEqual(registry.checkpoint_options("white"), before)

    def test_loader_uses_selected_snapshot_and_its_own_processors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = "act_bad_v1/checkpoints/020000/pretrained_model"
            checkpoint = root / source
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            features_path = root / "dataset_features.json"
            features_path.write_text(json.dumps({"action": {"shape": [6]}}))
            route = SimpleNamespace(name="20k", checkpoint=root / "unused", runner=Path("runner.py"),
                                    hub_repo="MikelRubi/act-up-down-bad", hub_path=source,
                                    hub_revision="pinned-revision", dataset_features=features_path,
                                    n_action_steps=25)
            runner = Mock()
            runner.load_policy.return_value = (Mock(), Mock(), Mock())
            with patch("huggingface_hub.snapshot_download", return_value=str(root)) as download, \
                    patch.object(app, "load_script_path", return_value=runner):
                loaded = app.load_route_components(route, "cpu")

            download.assert_called_once_with(route.hub_repo, revision="pinned-revision",
                                             allow_patterns=[f"{source}/*"])
            runner.load_policy.assert_called_once_with(checkpoint, "cpu", 25)
            loaded[1].reset.assert_called_once()
            self.assertEqual(loaded[-1], {"action": {"shape": [6]}})


if __name__ == "__main__":
    unittest.main()
