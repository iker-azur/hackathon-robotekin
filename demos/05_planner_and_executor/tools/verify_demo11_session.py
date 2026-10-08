"""Observe real localhost start/command/stop/restart, recording outcomes and timings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

HERE=Path(__file__).resolve().parents[1]
OUT=HERE/"out11"/"benchmarks"
PORT=18765
URL=f"http://127.0.0.1:{PORT}"


def request(path, body=None):
    payload=json.dumps(body).encode() if body is not None else None
    with urlopen(Request(URL+path,data=payload,headers={"Content-Type":"application/json"}),timeout=5) as reply:
        return json.load(reply)


def until(predicate, timeout=180):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=request("/api/state")
        if predicate(value): return value
        time.sleep(.1)
    raise TimeoutError("La sesión no alcanzó el estado esperado.")


def main():
    import sys
    sys.path.insert(0,str(HERE))
    import scene_config
    OUT.mkdir(parents=True,exist_ok=True)
    observations=[]; env=dict(os.environ,HF_HUB_OFFLINE="1",PYTHONIOENCODING="utf-8")
    with (OUT/"supervisor.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,str(HERE/"11_real_sorting_with_monitor.py"),"--headless","--no-viewer","--spread","0","--port",str(PORT)],stdout=log,stderr=log,env=env)
        original_config=None
        try:
            deadline=time.monotonic()+30
            while True:
                try: request("/api/health");break
                except OSError:
                    if process.poll() is not None or time.monotonic()>deadline: raise
                    time.sleep(.1)
            original_config=request("/api/config")["config"]
            for cycle in range(2):
                config=scene_config.default_config() if cycle==0 else scene_config.simple_config()
                if cycle:
                    for tray in config["containers"]:tray["size_mm"][2]=30
                request("/api/config",config)
                began=time.perf_counter(); start=request("/api/env",{"action":"start","seed":110+cycle})
                state=until(lambda s: any(c["id"]==start["id"] and c["state"] in ("SUCCESS","FAILED") for c in s["commands"]))
                start_state=next(c for c in state["commands"] if c["id"]==start["id"])
                observation={"cycle":cycle,"start_s":time.perf_counter()-began,"start":start_state}
                if start_state["state"]!="SUCCESS": raise RuntimeError(start_state["note"])
                # An unknown colour must finish visibly, before any task motion.
                bad=request("/api/command",{"text":"pon la bola magenta en la caja roja"})
                invalid=until(lambda s:any(c["id"]==bad["id"] and c["state"] in ("NEEDS_CLARIFICATION","FAILED") for c in s["commands"]))
                observation["invalid"]=next(c for c in invalid["commands"] if c["id"]==bad["id"])
                if observation["invalid"]["state"]!="NEEDS_CLARIFICATION":
                    raise RuntimeError("La orden desconocida no produjo una aclaración.")
                text="pon la bola roja en la caja roja" if cycle==0 else "pon el cubo azul en Azul A"
                began=time.perf_counter();command=request("/api/command",{"text":text})
                ping=[]
                for _ in range(15):
                    t=time.perf_counter();request("/api/health");ping.append((time.perf_counter()-t)*1000);time.sleep(.1)
                completed=until(lambda s:any(c["id"]==command["id"] and c["state"] in ("SUCCESS","PARTIAL_SUCCESS","FAILED","NEEDS_CLARIFICATION") for c in s["commands"]))
                observation.update(command_s=time.perf_counter()-began,health_ms=ping,
                    command=next(c for c in completed["commands"] if c["id"]==command["id"]),metrics=completed["metrics"])
                if observation["command"]["state"] not in {"SUCCESS","FAILED","PARTIAL_SUCCESS"}:
                    raise RuntimeError(observation["command"]["note"])
                # A physical failure must leave the environment available and explain why.
                if not observation["command"]["note"]:
                    raise RuntimeError("La orden terminó sin un motivo visible.")
                completed=until(lambda s:s["state"].get("phase")=="waiting" and s["state"].get("command_result",{}).get("id")==command["id"],timeout=10)
                observation["metrics"]=completed["metrics"]
                began=time.perf_counter();request("/api/env",{"action":"stop"})
                stopped=until(lambda s:s["state"].get("sim")=="stopped",timeout=30)
                observation.update(stop_s=time.perf_counter()-began,report=request("/api/report"),phase=stopped["state"]["phase"])
                observations.append(observation)
                (OUT/"sessions.json").write_text(json.dumps(observations,ensure_ascii=False,indent=2),encoding="utf-8")
            print(json.dumps([{k:v for k,v in row.items() if k not in ("report",)} for row in observations],ensure_ascii=False,indent=2),flush=True)
        except Exception:
            try:
                (OUT/"session_failure.json").write_text(json.dumps(request("/api/state"),ensure_ascii=False,indent=2),encoding="utf-8")
            except Exception: pass
            raise
        finally:
            try:
                request("/api/env",{"action":"stop"});until(lambda s:s["state"].get("sim")=="stopped",timeout=20)
                if original_config is not None:request("/api/config",original_config)
            except Exception: pass
            if process.poll() is None:
                if os.name=="nt":subprocess.run(["taskkill","/F","/T","/PID",str(process.pid)],capture_output=True)
                else:process.terminate()
            process.wait(timeout=5)


if __name__=="__main__":main()
