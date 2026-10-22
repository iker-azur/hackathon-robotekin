"""Reserve non-overlapping placements from declared cavity and localization error."""
from collections import defaultdict
import math

import real_world


def allocate(tasks, premise, items, occupied=None):
    by_name = {i.prim: i for i in items}
    groups = defaultdict(list)
    for task in tasks:
        if task.place is not None:
            groups[premise[task.place]].append((premise[task.pick], task))
    assigned = {}
    occupied = occupied or {}
    for destination, requested in groups.items():
        tray = by_name[destination]
        width = max(by_name[name].size[0] for name, _ in requested)
        depth = max(by_name[name].size[1] for name, _ in requested)
        # Reserve space for the camera error on both sides, plus a physical separation.
        gap = .003
        nx = math.floor((tray.inner[0] - 2*real_world.PERCEPTION_ERROR_M + gap) / (width+gap) + 1e-8)
        ny = math.floor((tray.inner[1] - 2*real_world.PERCEPTION_ERROR_M + gap) / (depth+gap) + 1e-8)
        slots = [((x-(nx-1)/2)*(width+gap), (y-(ny-1)/2)*(depth+gap)) for y in range(max(0, ny)) for x in range(max(0, nx))]
        existing = [(name, offset) for name, (bin_id, offset) in occupied.items() if bin_id == destination]
        for name, task in requested:
            candidates = [slot for slot in slots if all(
                previous == name or abs(slot[0]-offset[0]) >= (width+by_name[previous].size[0])/2+gap-1e-8
                or abs(slot[1]-offset[1]) >= (depth+by_name[previous].size[1])/2+gap-1e-8
                for previous, offset in existing)]
            if not candidates:
                raise ValueError(f"No queda espacio seguro en {destination} para todas las piezas. Añade un recipiente o amplía su cavidad.")
            slot = candidates[0]
            assigned[name] = (destination, slot)
            existing.append((name, slot))
    return assigned
