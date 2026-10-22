"""Short tests/static checks or a fixture-only UI screenshot; never train models."""

import os
from pathlib import Path
import sys
import tempfile
import uuid

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent


def run_tests(phase):
    import pytest
    scratch = HERE / "verification"
    scratch.mkdir(exist_ok=True)
    tempfile.tempdir = str(scratch)
    # Windows Python 3.11+ gives mode 0700 a user-only ACL. In the managed
    # sandbox that excludes the process's restricted token. Inherit the existing
    # workspace ACL ONLY for this runner's newly created temporary directories.
    original = os.mkdir
    def mkdir(path, mode=0o777, *, dir_fd=None):
        if os.name == "nt" and mode == 0o700 and Path(path).resolve().is_relative_to(scratch.resolve()):
            mode = 0o777
        return original(path, mode, dir_fd=dir_fd)
    os.mkdir = mkdir
    return pytest.main([str(DEMO / phase / "tests"), "-q", "-p", "no:cacheprovider", "--tb=short",
                        "--basetemp", str(scratch / (phase + "_" + uuid.uuid4().hex[:8]))])


def screenshot():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0,str(DEMO / "phase_2/src"))
    import cv2
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QImage, QFontDatabase
    from roboqc.ui.main_window import MainWindow
    from roboqc.ui.workstation import WorkstationWindow
    from roboqc.ui.camera_worker import CameraWorker
    from roboqc.industrial.voice import understand, to_legacy
    from roboqc.vision.config import load_config
    from roboqc.vision.calibration import WorkspaceCalibration
    from roboqc.vision.detector import Detection
    from roboqc.vision.overlay import draw_overlay
    # No microphone enumeration, model loading or camera acquisition.
    MainWindow._load_engine = lambda self: None
    MainWindow._populate_mics = lambda self: None
    MainWindow._populate_engines = lambda self: None
    WorkstationWindow._load_model = lambda self: None
    WorkstationWindow._populate_mics = lambda self: None
    WorkstationWindow._populate_engines = lambda self: None
    CameraWorker.start = lambda self: None
    app = QApplication([])
    # Use a bundled font for the sandbox screenshot; production stylesheet stays intact.
    font = Path("C:/Windows/Fonts/arial.ttf")
    if font.exists():
        QFontDatabase.addApplicationFont(str(font))
    import roboqc.ui.main_window as main_module
    import roboqc.ui.workstation as work_module
    main_module.STYLESHEET = main_module.STYLESHEET.replace('"Segoe UI", Arial','Arial')
    work_module.STYLESHEET = main_module.STYLESHEET
    scratch = HERE / "verification"
    scratch.mkdir(exist_ok=True)
    main = MainWindow(str(scratch / "absent_models"))
    response = understand("Pon las piezas buenas en la caja verde, las defectuosas tipo uno en la roja y las tipo dos en la azul")
    main._render_result(to_legacy(response,"Clasificacion por calidad"),[])
    main.show(); app.processEvents()
    main.grab().save(str(scratch / "voice_after.png"))
    cfg = load_config("__missing__")
    cfg["tracking"]["min_hits"] = 1
    cfg["industrial"] = {"enabled":False,"source_region":[50,50,300,300],"destinations":{"green":[400,50,60,60],"red":[400,150,60,60],"blue":[400,250,60,60]}}
    pixels = [[0,0],[639,0],[639,479],[0,479]]
    cal = WorkspaceCalibration.from_points(pixels,pixels,frame="robot")
    workstation = WorkstationWindow(0,cfg,cal,model_dir=str(scratch / "absent_models"))
    mask = np.zeros((480,640),np.uint8); mask[150:175,100:230] = 1
    det = Detection("red",.99,"industrial","bolt",.99,(100,150,130,25),(165,162),3250,0,.99,4,False,
        mask=mask,industrial_quality="good",grasp_candidates=[{"position_px":[165,162],"orientation_deg":0,"validated":False}],class_confidence=.99)
    workstation.world.update([det],__import__("time").monotonic())
    workstation.show_result(to_legacy(response,"Clasificacion por calidad"))
    frame = np.full((480,640,3),45,np.uint8); frame[mask > 0] = [40,40,210]
    cv2.putText(frame,"FIXTURE - NO CAMERA / NO MODEL",(30,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    snapshot = workstation.world.snapshot()
    view = cv2.cvtColor(draw_overlay(frame,snapshot.objects),cv2.COLOR_BGR2RGB)
    image = QImage(view.data,640,480,640*3,QImage.Format_RGB888).copy()
    workstation.show(); app.processEvents()
    workstation._on_camera_update(image,snapshot); app.processEvents()
    workstation.grab().save(str(scratch / "vision_after.png"))
    assert workstation.sorting.build(snapshot)[0]["status"] == "ready_for_simulation"
    workstation._simulate_step()
    assert "simulated_verification" in workstation.result_text.text()
    main.close(); workstation.close()
    print("Fixture screenshots and UI integration passed:",scratch)


if __name__ == "__main__":
    if sys.argv[1:] == ["screenshot"]:
        screenshot()
    elif sys.argv[1:] and sys.argv[1] in ("phase_1","phase_2"):
        raise SystemExit(run_tests(sys.argv[1]))
    else:
        raise SystemExit("Usage: verify_local.py phase_1|phase_2|screenshot")
