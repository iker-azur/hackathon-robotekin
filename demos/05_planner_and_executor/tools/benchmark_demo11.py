"""Repeatable CPU evaluation and observed Isaac capture, without changing executors.

Outputs go to out11/benchmarks. Run with the same Python environment as Demo 11.
  python tools/benchmark_demo11.py intent
  python tools/benchmark_demo11.py capture [Demo 11 arguments]
  python tools/benchmark_demo11.py perception
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import time

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
OUT = HERE / "out11" / "benchmarks"


def write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def item(value):
    return {k: value.get(k) for k in ("kind", "color", "quantity", "position", "place")}


def semantic(command):
    intent = command["intent"]
    return {"intent": intent, "items": [item(i) for i in command.get("items", [])],
            "place": command.get("place"), "sort_by": command.get("sort_by", "none"),
            "sort_filter": [item(i) for i in command.get("sort_filter", [])]}


def intent():
    import numpy as np
    from command_model_v2.command_model import CommandModel, plain
    from intent_parser import HybridParser
    data = HERE / "command_model_v2" / "data"
    train = {plain(json.loads(s)["input"]) for s in (data/"train_commands.jsonl").read_text(encoding="utf-8").splitlines()}
    rows = [json.loads(s) for s in (data/"test_real.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if plain(r["input"]) not in train]
    started = time.perf_counter()
    model = CommandModel()
    result = {"count": len(rows), "excluded_exact_train_overlaps": 136-len(rows),
              "load_s": time.perf_counter()-started, "backends": {}}
    # No catalog: this evaluates language extraction, independent of scene availability.
    for name, parser in [("model", model), ("rules", HybridParser()), ("hybrid", HybridParser(model))]:
        times, exact, intents, errors = [], 0, 0, []
        for row in rows:
            began = time.perf_counter(); got = parser.parse(row["input"])
            times.append((time.perf_counter()-began)*1000)
            expected = row["meta"]["command"]
            ok = semantic(expected) == semantic(got) and not got.get("errors")
            exact += ok; intents += expected["intent"] == got["intent"]
            if not ok: errors.append({"text": row["input"], "expected": semantic(expected), "got": got})
        result["backends"][name] = {"exact": exact, "intent": intents, "mean_ms": float(np.mean(times)),
                                     "p50_ms": float(np.percentile(times,50)), "p95_ms": float(np.percentile(times,95)), "errors": errors}
    write("intent.json", result)
    print(json.dumps({**result, "backends": {k:{a:b for a,b in v.items() if a!="errors"} for k,v in result["backends"].items()}}, indent=2), flush=True)


def capture(arguments):
    import cv2
    import numpy as np
    import isaac_io
    import telemetry
    spec = importlib.util.spec_from_file_location("benchmark_demo11_runtime", HERE/"11_real_sorting_with_monitor.py")
    demo = importlib.util.module_from_spec(spec); sys.modules[spec.name] = demo; spec.loader.exec_module(demo)
    args = demo.build_parser().parse_args(arguments)
    args.web = False; args.wait_for_start = False
    bus = telemetry.Bus(); timings = {}; stages = []; started = time.perf_counter()
    original_state = bus.set_state
    def state(**values):
        if "phase" in values: stages.append({"phase": values["phase"], "s":time.perf_counter()-started})
        original_state(**values)
        if values.get("phase") == "done":
            write("runtime.json", {"elapsed_s":time.perf_counter()-started,"stages":stages,
                                   "timings":timings,"snapshot":bus.snapshot()})
    bus.set_state = state
    original = demo.MonitoredRobot.look
    counter = 0
    def look(self, caption=""):
        nonlocal counter
        seen = original(self, caption)
        # Capture a paired frame with pose truth at the exact same instant; truth is
        # used only by the offline scorer, never supplied to the detector.
        if counter < 30:
            image, depth, camera = self._last_capture
            folder = OUT/"captures"; folder.mkdir(parents=True,exist_ok=True)
            stem = f"{args.seed}_{counter:03d}"
            cv2.imwrite(str(folder/(stem+".png")), image)
            if depth is not None: np.save(folder/(stem+".npy"),depth)
            truth = [{"id":i.prim,"kind":i.kind,"color":i.color,"container":i.container,
                      "size":list(i.size),
                      "position":self.env.scene[i.prim].data.root_pos_w[0].detach().cpu().tolist()} for i in self.layout]
            (folder/(stem+".json")).write_text(json.dumps({"camera":asdict(camera),"truth":truth,"caption":caption}),encoding="utf-8")
        counter += 1
        return seen
    demo.MonitoredRobot.look = look
    for owner, name in [(isaac_io,"capture"),(isaac_io,"capture_rgb"),(demo.MonitoredRobot,"look")]:
        function = getattr(owner,name)
        def timed(*a,_function=function,_name=name,**kw):
            began=time.perf_counter()
            try: return _function(*a,**kw)
            finally:
                row=timings.setdefault(_name,{"calls":0,"total_ms":0})
                row["calls"]+=1;row["total_ms"]+=(time.perf_counter()-began)*1000
        setattr(owner,name,timed)
    demo.main(args=args,bus=bus)


def perception():
    import cv2
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import perceive
    import scene_spec
    from scene import CameraModel
    reports = {}
    baseline_path=OUT/"baseline_perceive.py"
    baseline_path.write_bytes(subprocess.check_output(["git","show","979fd5b^:demos/05_planner_and_executor/perceive.py"],cwd=HERE))
    spec=importlib.util.spec_from_file_location("demo11_baseline_perceive",baseline_path)
    baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    for label, detector, mode in (("baseline",baseline,False),("legacy",perceive,False),("robust",perceive,True)):
        totals={"truth":0,"detections":0,"matched":0,"correct_kind":0,"xy_errors_mm":[],"latencies_ms":[],"frames":0}
        for path in sorted((OUT/"captures").glob("*.json")):
            meta=json.loads(path.read_text()); image=cv2.imread(str(path.with_suffix(".png")))
            camera=CameraModel.from_dict(meta["camera"])
            depth=np.load(path.with_suffix(".npy")) if path.with_suffix(".npy").exists() else None
            # Settled initial scene only: execution frames can legitimately be occluded
            # by the gripper or contain a piece inside its destination.
            if meta["caption"] != "the table": continue
            truth=meta["truth"]
            colors=sorted({t["color"] for t in truth})
            for brightness in (1., .7):
                frame=np.clip(image.astype(float)*brightness,0,255).astype(np.uint8)
                for repeat in range(6):
                    began=time.perf_counter()
                    parameters=dict(camera=camera,depth=depth,plane_z=scene_spec.TABLE_TOP_Z,looking_for=colors)
                    if detector is perceive:parameters["robust"]=mode
                    seen=detector.perceive(frame,**parameters)
                    elapsed=(time.perf_counter()-began)*1000
                    if repeat: totals["latencies_ms"].append(elapsed)
                detected=seen.everything; totals["frames"]+=1;totals["truth"]+=len(truth);totals["detections"]+=len(detected)
                cost=np.full((len(truth),len(detected)),1e6)
                for i,t in enumerate(truth):
                    for j,d in enumerate(detected):
                        if t["color"]==d.color and t["container"]==(d in seen.containers):
                            xy=d.aim_xy if mode and d.aim_xy is not None else scene_spec.aim_point(d.pos_m, container=t["container"])
                            cost[i,j]=np.linalg.norm(np.array(xy)-t["position"][:2])
                if cost.size:
                    for i,j in zip(*linear_sum_assignment(cost)):
                        if cost[i,j] <= .04:
                            totals["matched"]+=1; totals["correct_kind"]+=truth[i]["kind"]==detected[j].kind
                            totals["xy_errors_mm"].append(float(cost[i,j]*1000))
        totals["mean_ms"]=float(np.mean(totals["latencies_ms"])) if totals["latencies_ms"] else None
        totals["mean_xy_error_mm"]=float(np.mean(totals["xy_errors_mm"])) if totals["xy_errors_mm"] else None
        reports[label]=totals
    write("perception.json",reports);print(json.dumps(reports,indent=2),flush=True)


def extensions():
    """Freeze additional scene-vocabulary cases before running any parser on them.

    This is a scene extension test, not a claim of semantic independence from train.
    """
    import scene_config
    from intent_parser import HybridParser
    from command_model_v2.command_model import CommandModel
    cfg, parts = scene_config.load()
    cfg["containers"][0]["name"] = "Destino Norte"
    cfg["containers"][1]["name"] = "Destino Sur"
    catalog = scene_config.catalog(cfg, parts)
    destination = cfg["containers"][0]
    cases = []
    for language, template in [("es", "pon {n} bolas rojas en Destino Norte"),
                               ("en", "put {n} red balls into Destino Norte"),
                               ("eu", "hartu {n} bola gorri eta sartu kutxa Destino Norte")]:
        for number in range(1, 7):
            cases.append({"text": template.format(n=number), "lang":language,"quantity":number,"refuse":False})
    for language, sentences in {
        "es": ["pon la bola magenta en Destino Norte", "pon la bola roja en la caja fantasma", "no cojas la bola roja", "pon 0 bolas rojas en Destino Norte"],
        "en": ["put the magenta ball into Destino Norte", "put the red ball into the phantom box", "don't grab the red ball", "put 0 red balls into Destino Norte"],
        "eu": ["hartu bola magenta eta sartu kutxa Destino Norte", "hartu bola gorri eta sartu kutxa fantasma", "ez hartu bola gorri", "hartu 0 bola gorri eta sartu kutxa Destino Norte"]
    }.items():
        cases.extend({"text":s,"lang":language,"refuse":True} for s in sentences)
    write("extension_cases.json",cases)
    model=CommandModel(); result={}
    for name, parser in [("model",model),("rules",HybridParser(catalog=catalog)),("hybrid",HybridParser(model,catalog))]:
        correct=0; failures=[]; refused=0; elapsed=[]
        for case in cases:
            started=time.perf_counter();got=parser.parse(case["text"]);elapsed.append((time.perf_counter()-started)*1000)
            rejected=bool(got.get("errors")) or got["intent"] in {"not_doable","off_topic"}
            if case["refuse"]:
                ok=rejected;refused+=rejected
            else:
                items=got.get("items",[]);place=got.get("place") or {}
                ok=(not rejected and got["intent"]=="pick_and_place" and len(items)==1
                    and items[0]["kind"]=="ball" and items[0]["color"]=="red"
                    and items[0]["quantity"]==case["quantity"] and place.get("entity_id")==destination["id"])
            correct+=bool(ok)
            if not ok: failures.append({"case":case,"got":got})
        result[name]={"correct":correct,"total":len(cases),"safe_refusals":refused,"refusal_cases":12,
                      "mean_ms":sum(elapsed)/len(elapsed),"failures":failures}
    write("extensions.json",result)
    print(json.dumps({k:{a:b for a,b in v.items() if a!="failures"} for k,v in result.items()},indent=2),flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    mode=sys.argv[1]
    {"intent":intent,"extensions":extensions,"perception":perception,"capture":lambda:capture(sys.argv[2:])}[mode]()
