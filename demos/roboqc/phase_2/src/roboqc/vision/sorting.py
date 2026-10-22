"""Sorting plan and explicit simulated execution. No robot-library imports."""

import copy
import time
import math

from roboqc.industrial.voice import Dialogue


def inside(point, region):
    if region is None:
        return False
    x,y,w,h = region
    return w > 0 and h > 0 and x <= point[0] < x+w and y <= point[1] < y+h


class SortingCoordinator:
    def __init__(self, settings=None, calibration=None):
        self.dialogue = Dialogue()
        self.settings = settings or {}
        for rectangle in [self.settings.get("source_region"),*self.settings.get("destinations",{}).values()]:
            if rectangle is not None and (len(rectangle) != 4 or not all(math.isfinite(v) for v in rectangle) or min(rectangle[2:]) <= 0):
                raise ValueError("Box regions must be finite [x,y,width,height] rectangles")
        self.calibration = calibration
        self.completed = set()
        self.events = []
        self.plan = []

    def accept(self, response):
        if response["status"] != "valid":
            self.dialogue.running = False
            return
        command = response["command"]
        intent = command["intent"]
        if intent == "sort":
            self.dialogue.active = copy.deepcopy(command)
            self.dialogue.running = True
            self.completed.clear()
        elif intent in ("stop","cancel"):
            self.dialogue.running = False
            if intent == "cancel":
                self.dialogue.active = None
        elif intent == "resume" and self.dialogue.active is not None:
            self.dialogue.running = True

    def build(self, snapshot):
        command = self.dialogue.active
        self.plan = []
        if not command:
            return []
        rules = {r["category"]:r["destination"] for r in command["rules"]}
        for obj in snapshot.objects:
            if obj.object_id in self.completed:
                continue
            category = obj.industrial_quality if command["mode"] == "quality" else obj.color
            destination = rules.get(category)
            reason = None
            if not self.dialogue.running:
                reason = "classification_paused"
            elif time.monotonic()-snapshot.timestamp > 2.0:
                reason = "stale_camera_snapshot"
            elif obj.detection_status != "VISIBLE":
                reason = "not_fully_visible"
            elif obj.industrial_quality is None:
                reason = "legacy_detection_not_industrial"
            elif not inside(obj.position_px,self.settings.get("source_region")):
                reason = "source_region_not_configured_or_outside"
            elif not destination:
                reason = "unknown_category_or_no_rule"
            elif not obj.grasp_candidates:
                reason = "no_grasp_candidate"
            elif self.settings.get("destinations",{}).get(destination) is None:
                reason = "destination_not_configured"
            grasp = copy.deepcopy(obj.grasp_candidates[0]) if obj.grasp_candidates else None
            target = None
            region = self.settings.get("destinations",{}).get(destination)
            if region and self.calibration:
                target = self.calibration.describe(region[0]+region[2]/2,region[1]+region[3]/2)
            if grasp and self.calibration:
                grasp["workspace"] = self.calibration.describe(*grasp["position_px"])
                if grasp["workspace"] is not None:
                    try:
                        grasp["workspace_orientation_deg"] = self.calibration.orientation_to_world(*grasp["position_px"],grasp.get("orientation_deg"))
                    except ValueError:
                        grasp["workspace_orientation_deg"] = None
            if not reason and (not grasp.get("workspace") or grasp["workspace"]["frame"] != "robot" or grasp["workspace"]["units"] != "mm"):
                reason = "robot_frame_calibration_required"
            if not reason and (not target or target["frame"] != "robot" or target["units"] != "mm"):
                reason = "destination_outside_robot_calibration"
            self.plan.append({"object_id":obj.object_id,"category":category,"destination":destination,
                "grasp":grasp,"destination_workspace":target,"status":"blocked" if reason else "ready_for_simulation", "reason":reason,
                "snapshot_timestamp":snapshot.timestamp})
        return copy.deepcopy(self.plan)

    def simulate_one(self, snapshot):
        # Rebuild immediately: never act on a plan cached before a stop or scene change.
        plan = self.build(snapshot)
        item = next((i for i in plan if i["status"] == "ready_for_simulation"),None)
        if item is None:
            return {"status":"blocked", "plan":plan}
        for stage in ["action_requested","simulated_result","simulated_verification"]:
            self.events.append({"event":stage,"object_id":item["object_id"],"destination":item["destination"],
                "timestamp":time.monotonic(),"simulated":True,"physical_success":None})
        self.completed.add(item["object_id"])
        return {"status":"simulated", "item":item,"events":self.events[-3:]}
