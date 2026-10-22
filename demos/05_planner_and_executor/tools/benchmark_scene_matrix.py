"""Run a repeatable pick-and-place matrix for all shapes, cameras and scene variants.

Validate the generated scenes without Isaac:
  python tools/benchmark_scene_matrix.py --validate-only

Run it from the project's Isaac environment after closing other Demo 11/Isaac sessions:
  python tools/benchmark_scene_matrix.py --repetitions 3 --headless

Each scenario is a separate Isaac process so camera calibration, scene loading and task results
are isolated. The runner refuses to start if the Demo 11 dashboard is already listening, avoiding
an accidental second Kit process beside the live demonstration.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "out11" / "benchmarks"
sys.path.insert(0, str(HERE))
KINDS = ("cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism")
COMMANDS_ES = {
    "cube": "mueve el cubo azul a la caja roja",
    "ball": "mueve la bola azul a la caja roja",
    "cylinder": "mueve el cilindro azul a la caja roja",
    "triangular_prism": "mueve el prisma triangular azul a la caja roja",
    "hexagonal_prism": "mueve el prisma hexagonal azul a la caja roja",
}
CAMERAS = ("top", "front", "side")
VARIANTS = ("contained", "open")
LAYOUTS = ("single", "cluster")
REGIONS = ("central", "forward_pick")
VALIDATED_TARGET_PROFILES_MM = {
    "cube": {20},
    "ball": {20},
    "cylinder": {20, 30},
    "triangular_prism": {20, 30},
    "hexagonal_prism": {20},
}


def scene_config(kind: str, repetition: int, layout: str = "single",
                 camera: str | None = None, size_mm: int = 30,
                 region: str = "central") -> dict:
    """Build a valid, camera-visible target scene with deterministic pose variation."""
    if kind not in KINDS:
        raise ValueError(f"Unsupported object geometry: {kind}")
    if layout not in LAYOUTS:
        raise ValueError(f"Unsupported scene layout: {layout}")
    if not 6 <= size_mm <= 36:
        raise ValueError("Object size must be between 6 and 36 mm")
    if region not in REGIONS:
        raise ValueError(f"Unsupported test region: {region}")
    import scene_config as catalog

    cfg = catalog.default_config()
    # Matrix poses include rotated footprints that extend past the normal editor's shallow
    # staging strip. This scenario-specific rectangle still lies on the table and its object
    # points remain in the measured robot reach annulus.
    if region == "forward_pick":
        # Strict-contact trials showed that the stock setup's inner pick pose is not
        # reliably reachable once the tool's measured body-to-aperture offset is applied.
        # This measured rectangle keeps the complete editor bounds inside the reach annulus
        # while shifting the target and tray forward/right to the physical ball success pose.
        cfg["working_area"].update(x_m=.420, width_mm=240.0,
                                   y_m=-.460, depth_mm=190.0)
    else:
        cfg["working_area"]["x_m"] = .405
        cfg["working_area"]["width_mm"] = 220.0
        cfg["working_area"]["y_m"] = -.455
        cfg["working_area"]["depth_mm"] = 200.0
    if layout == "cluster" and region == "central":
        # The central cluster fixture stays within the arm's full reach rectangle after adding
        # two distractors on the far side of the source-to-tray path.
        cfg["working_area"]["x_m"] = .396
        cfg["working_area"]["width_mm"] = 222.0
        cfg["working_area"]["y_m"] = -.430
        cfg["working_area"]["depth_mm"] = 250.0
    box = next(row for row in cfg["containers"] if row["color"] == "red")
    # The old fixture put the tray beside the robot base. The polar reach check accepted it,
    # but the SO-101's side reach saturated during carry. Put it farther forward, where the
    # measured IK can reach, while keeping the fixture's colored source visibly separate.
    # Keep the source well clear of the destination and put both in poses with
    # demonstrated physical grasps. The previous "central" fixture used x=.348,
    # y=-.400 for the source, which passed the radial reach check but produced
    # 150-220 mm aperture residuals in clustered physical trials.
    box["x"], box["y"] = ((.480, -.4095) if region == "forward_pick"
                           else (.473, -.4095))
    box["size_mm"] = [84.0, 84.0, 30.0]
    box["cavity_mm"] = [70.0, 70.0]
    box["wall_mm"] = 7.0
    if layout == "cluster":
        # Keep the destination clear of the nearby, same-colour distractor pair.
        box["x"], box["y"] = ((.480, -.4095) if region == "forward_pick"
                               else (.473, -.4095))
    cfg["containers"] = [box]
    # Small deterministic XY changes test repeatability while seeded randomization remains
    # bounded around a pick pose with margin from the inner arm singularity.
    dx = (repetition % 3) * .004
    # Use distinct source and destination colours. Cluster layouts add two nearby green balls
    # so object association is exercised independently of the target.
    target_base_x = ((.370 if layout == "cluster" else .380)
                     if region == "forward_pick" else .380)
    # Both matrix regions now use a physically demonstrated source pose; the
    # distinct region controls layout span and seeded perturbation, not reachability.
    target_y = -.400
    target_x = target_base_x + dx
    cfg["objects"] = [{
        "id": "matrix_piece", "kind": kind, "color": "blue", "count": 1,
        "size_mm": size_mm, "height_mm": max(6, round(size_mm * 0.8)),
        "x": target_x, "y": target_y,
        "yaw_deg": 0.0 if layout == "cluster" else float((repetition % 3) * 30),
    }]
    if layout == "cluster":
        # Keep the same-colour pair visible in the same workspace while outside the robot's
        # target approach and carry corridors. Previous fixtures put distractors on either side
        # of that swept path and physically displaced the target before/while closing.
        if region == "central":
            # Keep the 40 mm same-colour pair inside the SO-101 tested reach and outside the
            # target-to-tray approach/carry corridor. The previous y=-.320 pair was beyond the
            # near-side radial workspace, so a nominally valid stress scene contained distractors
            # the robot could never pick up.
            green_a_xy, green_b_xy = ((.300, -.390), (.300, -.430))
            box["x"], box["y"] = .465, -.4095
        else:
            # Both distractors must remain outside the 205 mm measured pick envelope;
            # the previous first ball sat at 197 mm and made this nominal stress scene
            # physically unreachable before perception or grasping could be evaluated.
            green_a_xy, green_b_xy = ((.485, -.480), (.520, -.480))
        cfg["objects"].extend([
            {"id": "near_green_a", "kind": "ball", "color": "green", "count": 1,
             "size_mm": 30, "x": green_a_xy[0], "y": green_a_xy[1], "yaw_deg": 0},
            {"id": "near_green_b", "kind": "ball", "color": "green", "count": 1,
             "size_mm": 30, "x": green_b_xy[0], "y": green_b_xy[1], "yaw_deg": 0},
        ])
    canonical, items = catalog.build(cfg)
    expected_items = 2 if layout == "single" else 4
    if len(items) != expected_items or not any(item.container for item in items):
        raise ValueError(f"Matrix scene for {kind} did not produce one part and one receptacle")
    return canonical


def command_for(kind: str) -> str:
    """A direct Spanish order so every scenario exercises perception and manipulation."""
    try:
        return COMMANDS_ES[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported object geometry: {kind}") from exc


def grasp_profile_kind_for(kind: str, size_mm: int) -> str | None:
    """Scope a validated contact-pad profile to the requested part in mixed scenes."""
    if size_mm in VALIDATED_TARGET_PROFILES_MM.get(kind, set()):
        return kind
    return None


def capture_passed(returncode: int, report: dict | None) -> bool:
    """A scenario passes only when the task and placement verifier report success.

    Initial-frame detection is reported separately: a bounded re-look may correctly recover
    from a missed first frame, and should not turn a verified placement into a task failure.

    The part must also have been carried by friction: at least one physical grasp and no
    simulator-assisted (kinematically attached) grasp. A report that does not carry both
    counters cannot prove that, so it does not pass.
    """
    if returncode != 0 or not isinstance(report, dict):
        return False
    state = report.get("snapshot", {}).get("state", {})
    physical, assisted = state.get("real_grasps"), state.get("kinematic_grasps")
    return (state.get("phase") == "done" and state.get("status") == "SUCCESS"
            and state.get("completed_tasks") == 1 and state.get("failed_tasks") == 0
            and state.get("verified", 0) >= 1
            and isinstance(physical, int) and physical >= 1
            and isinstance(assisted, int) and assisted == 0)


def report_exit_code(report: dict) -> int:
    """Incomplete campaigns are not successful, even when every finished trial passed."""
    complete = report.get("total") == report.get("expected_total")
    return 0 if complete and report.get("failed", 0) == 0 else 2


def score_linked_detections(truth: list[dict], linked: list[dict]) -> dict:
    """Score the catalog-associated observations used by the planner, including duplicate IDs."""
    import numpy as np

    truth_by_id = {item["id"]: item for item in truth}
    errors, correct_kind, matched = [], 0, 0
    target_detected = False
    for expected in truth:
        candidates = [row for row in linked if row.get("entity_id") == expected["id"]]
        if len(candidates) != 1:
            continue
        actual = candidates[0]
        point = actual.get("aim_xy") or actual.get("position", [])[:2]
        if len(point) < 2:
            continue
        error = float(np.linalg.norm(np.asarray(point, dtype=float)
                                     - np.asarray(expected["position"][:2], dtype=float)))
        if (error > .025 or actual.get("color") != expected["color"]
                or actual.get("container") != expected["container"]):
            continue
        matched += 1
        errors.append(error * 1000)
        kind_ok = actual.get("kind") == expected["kind"]
        correct_kind += int(kind_ok)
        if not expected["container"] and expected["color"] == "blue":
            target_detected = kind_ok
    used_ids = [row.get("entity_id") for row in linked if row.get("entity_id") in truth_by_id]
    false_positives = len(linked) - len(set(used_ids))
    missed = max(0, len(truth) - matched)
    max_error = max(errors) if errors else None
    passed = (matched == len(truth) and correct_kind == len(truth)
              and false_positives == 0 and target_detected)
    return {"passed": passed,
            "system": {"truth_count": len(truth), "detection_count": len(linked),
                       "matched_ids": matched, "correct_kind": correct_kind,
                       "missed": missed, "false_positives": false_positives,
                       "mean_xy_error_mm": (round(float(np.mean(errors)), 2)
                                            if errors else None),
                       "max_xy_error_mm": round(max_error, 2) if max_error is not None else None},
            "target_detected_within_25mm": target_detected}


def perception_geometry_priors(truth: list[dict]) -> dict:
    """Return the same configured size bounds the live Demo 11 perception receives.

    The scene-matrix scorer re-runs perception over its saved RGB-D frame. Omitting the
    configured footprint bounds changes ``split_touching`` decisions and makes the
    reported recall differ from the detector used by the planner, especially for compact
    same-colour clusters. Keep this derivation aligned with ``MonitoredRobot.look``.
    """
    objects = [row for row in truth if not row.get("container", False)]
    sizes = [tuple(float(value) for value in row.get("size", (.03, .03, .03)))
             for row in objects]
    if not sizes:
        return {"smallest_part_m": None, "largest_part_m": None,
                "largest_part_footprint_m2": None, "object_bounds_by_color": {}}
    smallest_part = min(min(size[:2]) for size in sizes)
    largest_part = max(max(size[:2]) for size in sizes)
    largest_footprint = max(size[0] * size[1] for size in sizes)
    bounds_by_color = {}
    for row, size in zip(objects, sizes):
        extent = max(size[:2])
        footprint = size[0] * size[1]
        previous = bounds_by_color.get(row["color"])
        if previous is None or footprint > previous[1]:
            bounds_by_color[row["color"]] = (extent, footprint)
    return {"smallest_part_m": smallest_part,
            "largest_part_m": largest_part,
            "largest_part_footprint_m2": largest_footprint,
            "object_bounds_by_color": bounds_by_color}


def perception_work_bounds(truth: list[dict], metadata: dict) -> dict:
    """Return the configured live ROI bounds, or infer a conservative legacy ROI.

    ``workspace_roi`` is part of perception, so replaying a cluster frame with the
    checkout's default rectangle can hide objects that were validly placed in that
    run's customized work area. New captures persist the exact bounds; older captures
    get an image-scoring rectangle that covers every truth footprint plus the ROI pad.
    """
    area = metadata.get("work_area") or {}
    work_x, work_y = area.get("work_x"), area.get("work_y")
    if (isinstance(work_x, (list, tuple)) and len(work_x) == 2
            and isinstance(work_y, (list, tuple)) and len(work_y) == 2):
        return {"work_x": tuple(float(value) for value in work_x),
                "work_y": tuple(float(value) for value in work_y)}
    footprints = []
    for item in truth:
        pos = item.get("position")
        size = item.get("size")
        if not pos or not size or len(pos) < 2 or len(size) < 2:
            continue
        footprints.append((float(pos[0]), float(pos[1]),
                           abs(float(size[0])) / 2.0, abs(float(size[1])) / 2.0))
    if not footprints:
        import scene_spec
        return {"work_x": tuple(scene_spec.WORK_X), "work_y": tuple(scene_spec.WORK_Y)}
    pad = .025
    return {"work_x": (min(x - hx for x, _y, hx, _hy in footprints) - pad,
                       max(x + hx for x, _y, hx, _hy in footprints) + pad),
            "work_y": (min(y - hy for _x, y, _hx, hy in footprints) - pad,
                       max(y + hy for _x, y, _hx, hy in footprints) + pad)}


def score_initial_capture(seed: int, started_ns: int) -> dict | None:
    """Score raw vision and catalog-linked observations against simulator pose truth."""
    import cv2
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import perceive
    import scene_spec
    from scene import CameraModel

    meta_path = OUT / "captures" / f"{seed}_000.json"
    image_path = meta_path.with_suffix(".png")
    if not meta_path.exists() or meta_path.stat().st_mtime_ns < started_ns:
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    image = cv2.imread(str(image_path))
    if image is None or meta.get("caption") != "the table":
        return None
    depth_path = meta_path.with_suffix(".npy")
    depth = np.load(depth_path) if depth_path.exists() else None
    truth = meta.get("truth", [])
    if not truth:
        return None
    geometry_priors = perception_geometry_priors(truth)
    work_bounds = perception_work_bounds(truth, meta)
    previous_work_x, previous_work_y = scene_spec.WORK_X, scene_spec.WORK_Y
    scene_spec.WORK_X, scene_spec.WORK_Y = work_bounds["work_x"], work_bounds["work_y"]
    try:
        seen = perceive.perceive(
            image, source="scene-matrix", camera=CameraModel.from_dict(meta["camera"]),
            plane_z=scene_spec.TABLE_TOP_Z,
            looking_for=sorted({item["color"] for item in truth}), depth=depth, robust=True,
            min_area_px=perceive.colors.min_area_for_object_size(
                geometry_priors["smallest_part_m"]),
            max_object_extent_m=geometry_priors["largest_part_m"],
            max_object_footprint_m2=geometry_priors["largest_part_footprint_m2"],
            object_bounds_by_color=geometry_priors["object_bounds_by_color"],
        )
    finally:
        scene_spec.WORK_X, scene_spec.WORK_Y = previous_work_x, previous_work_y
    detections = seen.everything
    cost = np.full((len(truth), len(detections)), 1e6, dtype=float)
    for row, expected in enumerate(truth):
        for col, actual in enumerate(detections):
            is_container = actual in seen.containers
            if expected["color"] != actual.color or expected["container"] != is_container:
                continue
            if actual.pos_m is None and actual.aim_xy is None:
                continue
            aim = (actual.aim_xy if actual.aim_xy is not None else
                   scene_spec.aim_point(actual.pos_m, container=is_container))
            cost[row, col] = float(np.linalg.norm(
                np.asarray(aim, dtype=float) - np.asarray(expected["position"][:2], dtype=float)))
    raw_matched, raw_correct_kind, raw_target_kind_correct, raw_target_kind_total = 0, 0, 0, 0
    raw_errors = []
    if cost.size:
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            error = float(cost[row, col])
            if error > .040:
                continue
            raw_matched += 1
            raw_errors.append(error * 1000)
            raw_correct_kind += int(detections[col].kind == truth[row]["kind"])
            if not truth[row].get("container", False):
                raw_target_kind_total += 1
                raw_target_kind_correct += int(detections[col].kind == truth[row]["kind"])

    # Rebuild the catalog association from this run's detector output. Cached live detections
    # describe the old binary and would conceal a segmentation regression or improvement.
    linked = []
    if cost.size:
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            if cost[row, col] > .040:
                continue
            expected, actual = truth[row], detections[col]
            is_container = actual in seen.containers
            position = list(actual.pos_m) if actual.pos_m is not None else expected["position"]
            aim = actual.aim_xy if actual.aim_xy is not None else position[:2]
            linked.append({
                "entity_id": expected["id"], "kind": expected["kind"],
                "color": actual.color, "container": is_container,
                "position": position, "aim_xy": list(aim),
            })
    linked_report = score_linked_detections(truth, linked)
    linked_report["raw_detector"] = {"truth_count": len(truth), "detection_count": len(detections),
                         "matched_within_40mm": raw_matched,
                         "correct_kind": raw_correct_kind,
                         "target_kind_correct": raw_target_kind_correct,
                         "target_kind_total": raw_target_kind_total,
                         "false_positives": max(0, len(detections) - raw_matched),
                         "mean_xy_error_mm": (round(float(np.mean(raw_errors)), 2)
                                              if raw_errors else None)}
    return linked_report


def _runtime_for(seed: int) -> Path:
    return OUT / f"runtime_{seed}.json"


def _dashboard_is_live(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=.25):
            return True
    except OSError:
        return False


def run(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--geometries", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--region", choices=REGIONS, default="central",
                        help="measured central fixture or forward pick pose for strict contact trials")
    parser.add_argument("--sizes-mm", nargs="+", type=int, default=[30],
                        help="object diameters/edge lengths; each must be in [6, 36] mm")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11500)
    parser.add_argument("--spread", type=float, default=.004,
                        help="maximum seeded XY perturbation in metres; defaults to 4 mm")
    parser.add_argument("--grip-stiffness", type=float, default=250.0,
                        help="gripper actuator stiffness for physical-grasp A/B trials")
    parser.add_argument("--close-command", type=float, default=None,
                        help="explicit gripper close target override in radians")
    parser.add_argument("--start-index", type=int, default=0,
                        help="resume at this zero-based cell; earlier rows are reconstructed from artifacts")
    parser.add_argument("--only-seeds", nargs="+", type=int,
                        help="run only these already-generated matrix seeds (useful for exact retries)")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--strict-physical", action="store_true",
                        help="disable automatic simulator grasp assistance in Demo 11 scenarios")
    parser.add_argument("--sim-assist-policy", choices=("preemptive", "after_physical_miss"),
                        default="preemptive",
                        help="when auto-assist is enabled, either attach before contact or attempt physical grasp first")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate scene generation and all expected cells without launching Isaac")
    parser.add_argument("--output", type=Path, default=OUT / "scene_matrix.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(arguments)
    if args.repetitions < 1:
        parser.error("--repetitions must be at least one")
    if args.close_command is not None and not 0.0 <= args.close_command <= 1.745:
        parser.error("--close-command must be within [0, 1.745] rad")
    if not 0.0 <= args.spread <= 0.020:
        parser.error("--spread must be between 0 and 0.020 metres")

    if any(size < 6 or size > 36 for size in args.sizes_mm):
        parser.error("--sizes-mm values must be between 6 and 36")
    cells = list(itertools.product(args.cameras, args.variants, args.geometries, args.layouts,
                                   args.sizes_mm, range(args.repetitions)))
    if args.start_index < 0 or args.start_index > len(cells):
        parser.error(f"--start-index must be between 0 and {len(cells)}")
    scenarios = []
    for index, (camera, variant, kind, layout, size_mm, repetition) in enumerate(cells):
        seed = args.seed + index
        config = scene_config(kind, repetition, layout, camera=camera, size_mm=size_mm,
                              region=args.region)
        scene_path = OUT / "scene_matrix" / (
            f"scene_{seed}_{camera}_{variant}_{kind}_{layout}_{size_mm}mm_{args.region}.json")
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        scene_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        scenarios.append({"camera": camera, "variant": variant, "kind": kind,
                          "layout": layout, "size_mm": size_mm,
                          "region": args.region, "repetition": repetition + 1, "seed": seed,
                          "scene_config": str(scene_path), "command": command_for(kind)})

    if args.only_seeds:
        by_seed = {scenario["seed"]: scenario for scenario in scenarios}
        missing_seeds = sorted(set(args.only_seeds) - set(by_seed))
        if missing_seeds:
            parser.error(f"--only-seeds contains seeds outside this matrix: {missing_seeds}")
        scenarios = [by_seed[seed] for seed in args.only_seeds]

    if args.validate_only:
        print(json.dumps({"validated": len(scenarios), "cameras": args.cameras,
                          "variants": args.variants, "geometries": args.geometries,
                          "layouts": args.layouts, "region": args.region,
                          "sizes_mm": args.sizes_mm,
                          "repetitions": args.repetitions}, ensure_ascii=False, indent=2))
        return 0
    if _dashboard_is_live(args.host, args.port):
        print(f"Demo 11 ya responde en {args.host}:{args.port}; no iniciaré otro proceso Isaac.",
              file=sys.stderr)
        return 3

    logs = OUT / "scene_matrix_logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    def save_checkpoint() -> None:
        report = {"seed": args.seed, "repetitions": args.repetitions,
                  "cameras": args.cameras, "variants": args.variants,
                  "geometries": args.geometries, "layouts": args.layouts,
                  "region": args.region, "sizes_mm": args.sizes_mm,
                  "grip_stiffness": args.grip_stiffness,
                  "close_command_override": args.close_command,
                  "total": len(results),
                  "expected_total": len(scenarios),
                  "complete": len(results) == len(scenarios),
                  "pending": max(0, len(scenarios) - len(results)),
                  "passed": sum(bool(row["passed"]) for row in results),
                  "failed": sum(not row["passed"] for row in results),
                  "first_frame_perception_passed": sum(
                      bool(row.get("initial_perception_passed")) for row in results),
                  "perception_recovered": sum(bool(row.get("perception_recovered"))
                                               for row in results),
                  "success_criteria": "one verified placement and zero failed tasks per scenario",
                  "perception_metric": "initial frame scored separately; a task may recover by re-looking",
                  "scenarios": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
        temp_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_output.replace(args.output)

    for index, scenario in enumerate(scenarios):
        seed = scenario["seed"]
        log_path = logs / (f"{seed}_{scenario['camera']}_{scenario['variant']}_"
                           f"{scenario['kind']}_{scenario['layout']}_"
                           f"{scenario['size_mm']}mm.log")
        runtime_path = _runtime_for(seed)
        if index < args.start_index:
            try:
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                runtime = None
            perception = score_initial_capture(seed, 0)
            # These rows are reconstructed only from completed runtime reports plus the saved
            # initial frame. This lets long matrices resume without treating missing artifacts
            # as a pass or rerunning already verified Isaac scenarios.
            if runtime is not None:
                runtime["perception"] = perception or {"passed": False, "error": "missing initial capture"}
            returncode = (0 if capture_passed(0, runtime)
                          and perception and perception.get("passed") else -1)
            row = {**scenario, "returncode": returncode, "elapsed_s": None,
                   "passed": capture_passed(returncode, runtime),
                   "runtime_report": str(runtime_path) if runtime else None,
                   "log": str(log_path) if log_path.exists() else None,
                   "error": "reconstructed from saved artifacts"}
            row["perception"] = perception
            row["initial_perception_passed"] = bool(perception and perception.get("passed"))
            row["perception_recovered"] = bool(row["passed"] and not row["initial_perception_passed"])
            if runtime:
                state = runtime.get("snapshot", {}).get("state", {})
                row.update(status=state.get("status"), completed_tasks=state.get("completed_tasks"),
                           failed_tasks=state.get("failed_tasks"), verified=state.get("verified"),
                           physical_grasps=state.get("real_grasps"),
                           simulator_assisted_grasps=state.get("kinematic_grasps"))
            results.append(row)
            continue
        argv = [sys.executable, str(HERE / "tools" / "benchmark_demo11.py"), "capture",
                "--scene-config", scenario["scene_config"], "--camera", scenario["camera"],
                "--variant", scenario["variant"], "--rounds", "1", "--seed", str(seed),
                "--spread", str(args.spread), "--text", scenario["command"], "--intent-backend", "rules",
                "--grip-stiffness", str(args.grip_stiffness),
                "--device", args.device, "--no-web", "--no-viewer",
                "--sim-assist-policy", args.sim_assist_policy]
        if scenario["layout"] == "cluster":
            argv.append("--allow-same-colour-clusters")
        if args.headless:
            argv.append("--headless")
        if args.strict_physical:
            argv.append("--strict-physical")
        profile_kind = grasp_profile_kind_for(scenario["kind"], scenario["size_mm"])
        if profile_kind is not None:
            # Scope validated contact pads to the resolved target so unrelated distractors
            # in clustered scenes do not silently force the grasp back to stock fingers.
            argv.extend(("--grasp-profile-kind", profile_kind))
        if args.close_command is not None:
            argv.extend(("--close-command", str(args.close_command)))
        started = time.perf_counter()
        started_ns = time.time_ns()
        returncode = -1
        error = ""
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                completed = subprocess.run(argv, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=600, check=False)
            returncode = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            runtime = (json.loads(runtime_path.read_text(encoding="utf-8"))
                       if runtime_path.stat().st_mtime_ns >= started_ns else None)
        except (OSError, json.JSONDecodeError):
            runtime = None
        perception = score_initial_capture(seed, started_ns)
        if runtime is not None:
            runtime["perception"] = perception or {"passed": False, "error": "missing initial capture"}
        row = {**scenario, "returncode": returncode,
               "elapsed_s": round(time.perf_counter() - started, 1),
               "passed": capture_passed(returncode, runtime),
               "runtime_report": str(runtime_path) if runtime else None,
               "log": str(log_path), "error": error}
        row["perception"] = perception
        row["initial_perception_passed"] = bool(perception and perception.get("passed"))
        row["perception_recovered"] = bool(row["passed"] and not row["initial_perception_passed"])
        if runtime:
            state = runtime.get("snapshot", {}).get("state", {})
            row.update(status=state.get("status"), completed_tasks=state.get("completed_tasks"),
                       failed_tasks=state.get("failed_tasks"), verified=state.get("verified"),
                       physical_grasps=state.get("real_grasps"),
                       simulator_assisted_grasps=state.get("kinematic_grasps"))
        results.append(row)
        save_checkpoint()
        print(f"{scenario['camera']} / {scenario['variant']} / {scenario['kind']} "
              f"/ {scenario['layout']} / {scenario['size_mm']} mm "
              f"{scenario['repetition']}/{args.repetitions}: "
              f"{'PASS' if row['passed'] else 'FAIL'} ({row.get('status', error or returncode)})",
              flush=True)

    save_checkpoint()
    report = json.loads(args.output.read_text(encoding="utf-8"))
    suffix = (f"; {report['pending']} pendientes" if report.get("pending") else "")
    print(f"\nRESULT {report['passed']}/{report['total']}{suffix}; report: {args.output}", flush=True)
    return report_exit_code(report)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    raise SystemExit(run())
