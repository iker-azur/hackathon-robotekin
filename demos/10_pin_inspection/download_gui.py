"""Small desktop launcher for the pin dataset downloader."""

import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


SCRIPT = Path(__file__).resolve().with_name("download_datasets.py")


class DownloadWindow:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.events = queue.Queue()
        root.title("Descargar datasets · Piezas con/sin perno")
        root.geometry("740x560")
        root.minsize(580, 460)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(8, weight=1)

        ttk.Label(frame, text="Imágenes y etiquetas de Hugging Face", font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))
        self.owner = tk.StringVar(value="MikelRubi")
        self.destination = tk.StringVar(value=str(SCRIPT.parent / "data"))
        ttk.Label(frame, text="Usuario:").grid(row=1, column=0, sticky="w", padx=(0, 10))
        self.owner_entry = ttk.Entry(frame, textvariable=self.owner)
        self.owner_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Label(frame, text="Guardar en:").grid(row=2, column=0, sticky="w", padx=(0, 10))
        self.path_entry = ttk.Entry(frame, textvariable=self.destination)
        self.path_entry.grid(row=2, column=1, sticky="ew", pady=4)
        self.browse = ttk.Button(frame, text="Elegir carpeta…", command=self.choose_folder)
        self.browse.grid(row=2, column=2, padx=(8, 0))
        self.with_pin_labels = tk.StringVar()
        self.without_pin_labels = tk.StringVar()
        ttk.Label(frame, text="Etiquetas adicionales opcionales, separadas por ;").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 4))
        ttk.Label(frame, text="Con perno:").grid(row=4, column=0, sticky="w", padx=(0, 10))
        self.with_pin_entry = ttk.Entry(frame, textvariable=self.with_pin_labels)
        self.with_pin_entry.grid(row=4, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Label(frame, text="Sin perno:").grid(row=5, column=0, sticky="w", padx=(0, 10))
        self.without_pin_entry = ttk.Entry(frame, textvariable=self.without_pin_labels)
        self.without_pin_entry.grid(row=5, column=1, columnspan=2, sticky="ew", pady=4)
        self.start_button = ttk.Button(frame, text="Descargar y separar en ConPerno / SinPerno", command=self.start)
        self.start_button.grid(row=6, column=0, columnspan=3, sticky="w", pady=12)
        self.status = tk.StringVar(value="Elige la carpeta de destino y pulsa Descargar.")
        ttk.Label(frame, textvariable=self.status, wraplength=670).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.log = scrolledtext.ScrolledText(frame, height=12, state="disabled", wrap="word")
        self.log.grid(row=8, column=0, columnspan=3, sticky="nsew")
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def choose_folder(self):
        folder = filedialog.askdirectory(parent=self.root, title="Carpeta de destino de los datasets")
        if folder:
            self.destination.set(folder)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_running(self, running):
        for widget in (self.owner_entry, self.path_entry, self.browse, self.start_button,
                       self.with_pin_entry, self.without_pin_entry):
            widget.configure(state="disabled" if running else "normal")
        if running:
            self.progress.start()
        else:
            self.progress.stop()

    def start(self):
        owner, destination = self.owner.get().strip(), self.destination.get().strip()
        if not owner or not destination:
            messagebox.showerror("Faltan datos", "Indica el usuario y la carpeta de destino.", parent=self.root)
            return
        folder = Path(destination).expanduser().resolve()
        if folder.exists() and not folder.is_dir():
            messagebox.showerror("Ruta incorrecta", "El destino debe ser una carpeta.", parent=self.root)
            return
        self.destination.set(str(folder))
        settings = {"owner": owner, "output": str(folder),
                    "with_pin_labels": [s.strip() for s in self.with_pin_labels.get().split(";") if s.strip()],
                    "without_pin_labels": [s.strip() for s in self.without_pin_labels.get().split(";") if s.strip()]}
        command = [sys.executable, "-X", "utf8", "-u", str(SCRIPT.with_name("download_worker.py"))]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8", errors="replace")
            self.process.stdin.write(json.dumps(settings))
            self.process.stdin.close()
        except OSError as exc:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            self.process = None
            messagebox.showerror("No se pudo iniciar", str(exc), parent=self.root)
            return
        self.set_running(True)
        self.status.set("Buscando datasets y descargando las imágenes y etiquetas reconocidas…")
        self.append_log(f"\nUsuario: {owner}\nDestino: {folder}\n")
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()

    def read_output(self, process):
        with process.stdout:
            for line in process.stdout:
                self.events.put(("log", line))
        self.events.put(("done", process.wait()))

    def poll(self):
        # All widget updates happen on Tk's main thread.
        for _ in range(200):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.append_log(value)
            else:
                self.process = None
                self.set_running(False)
                self.status.set("Descarga completada. Imágenes separadas en ConPerno y SinPerno según sus etiquetas." if value == 0 else
                                "La descarga no se completó. Revisa el registro y download_report.json, si se ha generado.")
        self.root.after(100, self.poll)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    DownloadWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
