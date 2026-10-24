"""Policy routing stays independent from the classifier and robot loop."""

import json
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

MODULE_PATH = Path(__file__).resolve().parents[1] / "policy_router.py"
SPEC = importlib.util.spec_from_file_location("policy_router", MODULE_PATH)
policy_router = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy_router
SPEC.loader.exec_module(policy_router)
PolicyRouteRegistry = policy_router.PolicyRouteRegistry


class PolicyRouteTests(unittest.TestCase):
    def test_both_classifier_results_can_select_the_current_test_policy(self):
        registry = PolicyRouteRegistry()

        pin = registry.resolve("PIN")
        no_pin = registry.resolve("NO PIN")

        self.assertEqual(pin.name, "up-down-bad")
        self.assertEqual(no_pin.name, "up-down-bad")
        self.assertEqual(pin.checkpoint, no_pin.checkpoint)
        self.assertEqual(pin.n_action_steps, 25)

    def test_route_config_can_map_labels_to_different_policies(self):
        config = {
            "classifier_routes": {"PIN": "good", "NO PIN": "bad"},
            "policies": {
                name: {"runner": "runs/policy.py", "checkpoint": f"runs/{name}",
                       "max_seconds": 10, "n_action_steps": 5, "max_relative_target": 5}
                for name in ("good", "bad")
            },
        }
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "routes.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            registry = PolicyRouteRegistry(path)

        self.assertEqual(registry.resolve("PIN").name, "good")
        self.assertEqual(registry.resolve("NO PIN").name, "bad")

    def test_unknown_classifier_result_has_no_implicit_policy(self):
        registry = PolicyRouteRegistry()

        with self.assertRaisesRegex(ValueError, "No post-classification policy"):
            registry.resolve("UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
