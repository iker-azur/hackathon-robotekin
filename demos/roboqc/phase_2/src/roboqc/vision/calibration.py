"""Optional pixel-to-workspace calibration.

A planar homography maps camera pixels to table coordinates. It is only built
from explicit correspondences supplied by the user (calibration JSON). Without a
file, world coordinates are None: pixel positions are never converted into
invented millimetres.

File format (config/calibration.json, values are examples):
    {"frame": "table", "units": "mm",
     "pixel_points": [[80, 60], [560, 60], [560, 420], [80, 420]],
     "world_points": [[0, 0], [300, 0], [300, 200], [0, 200]]}
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


class WorkspaceCalibration:
    def __init__(self, matrix: np.ndarray, frame: str, units: str):
        self.matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(self.matrix).all() or np.linalg.matrix_rank(self.matrix) < 3:
            raise ValueError("Matriz de calibracion invalida.")
        self.frame = frame
        self.units = units
        self.pixel_hull = None
        self.camera_matrix = None
        self.distortion = None
        self.robot_transform = None
        self.reprojection_error_px = None

    @classmethod
    def from_points(
        cls,
        pixel_points: Sequence[Sequence[float]],
        world_points: Sequence[Sequence[float]],
        frame: str = "table",
        units: str = "mm",
        camera_matrix=None,
        distortion=None,
        robot_transform=None,
        max_error_px: float = 2.0,
    ) -> "WorkspaceCalibration":
        if len(pixel_points) != len(world_points) or len(pixel_points) < 4:
            raise ValueError("La calibracion necesita al menos 4 pares pixel/mundo.")
        import cv2
        src = np.asarray(pixel_points, dtype=np.float64)
        dst = np.asarray(world_points, dtype=np.float64)
        if src.shape != (len(pixel_points),2) or dst.shape != src.shape or not np.isfinite(src).all() or not np.isfinite(dst).all():
            raise ValueError("Puntos de calibracion invalidos.")
        intrinsics = None
        if camera_matrix is not None:
            intrinsics = np.asarray(camera_matrix,dtype=np.float64).reshape(3,3)
            if distortion is None or not np.isfinite(intrinsics).all():
                raise ValueError("La correccion optica requiere intrinsecos y distorsion.")
            distortion = np.asarray(distortion,dtype=np.float64)
            if not np.isfinite(distortion).all():
                raise ValueError("Distorsion invalida.")
            src = cv2.undistortPoints(src.reshape(-1,1,2),intrinsics,distortion,P=intrinsics).reshape(-1,2)
        matrix, _ = cv2.findHomography(src, dst, 0)
        if matrix is None:
            raise ValueError("No se pudo calcular la homografia: puntos degenerados.")
        calibration = cls(matrix, frame, units)
        predicted = cv2.perspectiveTransform(dst.reshape(-1,1,2), np.linalg.inv(matrix)).reshape(-1,2)
        error = float(np.max(np.linalg.norm(predicted-src,axis=1)))
        if not np.isfinite(error) or error > max_error_px:
            raise ValueError(f"Error de reproyeccion excesivo: {error:.2f} px")
        calibration.pixel_hull = cv2.convexHull(src.astype(np.float32))
        calibration.camera_matrix = intrinsics
        calibration.distortion = distortion
        calibration.reprojection_error_px = error
        if robot_transform is not None:
            transform = np.asarray(robot_transform,dtype=np.float64).reshape(3,3)
            rotation = transform[:2,:2]
            if units != "mm" or not np.isfinite(transform).all() or not np.allclose(transform[2],[0,0,1]) or not np.allclose(rotation.T@rotation,np.eye(2),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1):
                raise ValueError("robot_transform debe ser rigida planar y usar mm.")
            calibration.robot_transform = transform
        return calibration

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        import cv2
        if not np.isfinite([px,py]).all():
            raise ValueError("Coordenadas de pixel invalidas.")
        if self.camera_matrix is not None:
            px,py = cv2.undistortPoints(np.array([[[px,py]]],dtype=np.float64),self.camera_matrix,self.distortion,P=self.camera_matrix)[0,0]
        if self.pixel_hull is not None and cv2.pointPolygonTest(self.pixel_hull,(float(px),float(py)),False) < 0:
            raise ValueError("Pixel fuera de la region calibrada.")
        x, y, w = self.matrix @ np.array([px, py, 1.0])
        if abs(w) < 1e-10:
            raise ValueError("Proyeccion degenerada.")
        if self.robot_transform is not None:
            x,y,w = self.robot_transform @ np.array([x/w,y/w,1.0])
        return float(x / w), float(y / w)

    def orientation_to_world(self, px, py, angle_deg):
        if angle_deg is None:
            return None
        a = np.radians(angle_deg)
        first = self.pixel_to_world(px,py)
        second = self.pixel_to_world(px+np.cos(a),py+np.sin(a))
        return float(np.degrees(np.arctan2(second[1]-first[1],second[0]-first[0])) % 180)

    def describe(self, px: float, py: float) -> Dict[str, Any]:
        """Return the world position with its coordinate frame and units."""
        try:
            x, y = self.pixel_to_world(px, py)
        except ValueError:
            return None
        return {"frame": "robot" if self.robot_transform is not None else self.frame, "units": self.units, "xy": [round(x, 2), round(y, 2)]}


def load_calibration(path: Optional[str]) -> Optional[WorkspaceCalibration]:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return WorkspaceCalibration.from_points(
        data["pixel_points"], data["world_points"],
        frame=data.get("frame", "table"), units=data.get("units", "mm"),
        camera_matrix=data.get("camera_matrix"), distortion=data.get("distortion"),
        robot_transform=data.get("robot_transform"), max_error_px=data.get("max_error_px",2.0),
    )
