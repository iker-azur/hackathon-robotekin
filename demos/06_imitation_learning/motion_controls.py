"""Editable movement controls and validation for the SO-101 pick/classify window."""

import math


# key, tab, label, default, suggested values, minimum, maximum
NUMERIC_FIELDS = [
    ("pick_seconds", "Recogida", "Duración máxima (s)", 20, (5, 10, 14, 20, 30, 60, 120), 0.1, 300),
    ("pick_step", "Recogida", "Paso máximo (° / % pinza)", 8, (0.5, 1, 2, 3, 5, 7, 8, 10, 15), 0.1, 30),
    ("pick_actions", "Recogida", "Acciones por predicción", None, (1, 5, 10, 15, 25, 30, 50, 90, 100), 1, 1000),
    ("first_gap", "Recogida", "Diferencia inicial máxima (° / %)", 60, (15, 30, 45, 60, 90, 120), 1, 180),
    ("min_pick", "Parada automática", "Tiempo mínimo antes de parar (s)", 5, (0, 2, 3, 5, 8, 10, 15), 0, 120),
    ("pose_window", "Parada automática", "Tiempo estable necesario (s)", 1.5, (0.5, 1, 1.5, 2, 3, 5), 0.1, 10),
    ("arm_motion", "Parada automática", "Movimiento tolerado del brazo (°)", 2, (0.5, 1, 2, 3, 5, 10), 0.1, 30),
    ("arm_error", "Parada automática", "Distancia al objetivo tolerada (°)", 6, (1, 2, 4, 6, 8, 10, 15, 30), 0.1, 90),
    ("gripper_motion", "Parada automática", "Movimiento tolerado de pinza (%)", 5, (1, 2, 3, 5, 8, 10, 20), 0, 100),
    ("control_hz", "Clasificación / control", "Frecuencia de control (Hz)", 30, (5, 10, 15, 20, 25, 30, 40, 50, 60), 1, 60),
    ("settle_seconds", "Clasificación / control", "Pausa de estabilización (s)", 0.5, (0, 0.25, 0.5, 1, 2, 3), 0, 20),
    ("classify_seconds", "Clasificación / control", "Tiempo de clasificación (s)", 1.5, (0.5, 1, 1.5, 2, 3, 5), 0.1, 20),
    ("pin_threshold", "Clasificación / control", "Umbral para decidir CON PIN (%)", 50, (30, 40, 50, 60, 70, 80, 90), 0, 100),
]
BOOL_FIELDS = [("auto_stop", "Parada automática", "Parar al alcanzar la pose", True)]
for destination, tab in (("green", "Caja verde"), ("white", "Caja blanca")):
    NUMERIC_FIELDS.extend([
        (f"{destination}_seconds", tab, "Duración máxima (s)", None, (5, 10, 15, 20, 30, 60, 120), 0.1, 300),
        (f"{destination}_step", tab, "Paso máximo (° / % pinza)", None, (0.5, 1, 2, 3, 5, 7, 10, 15), 0.1, 30),
        (f"{destination}_actions", tab, "Acciones por predicción", None, (1, 5, 10, 15, 25, 30, 50, 100), 1, 1000),
        (f"{destination}_height_cm", tab, "Elevación adicional (cm)", 3, (0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30), 0, 30),
        (f"{destination}_descent_mm", tab, "Descenso en corrección (mm/ciclo)", 2, (0, 0.5, 1, 2, 3, 5, 10), 0, 50),
        (f"{destination}_closed", tab, "Pinza: inicio de apertura (%)", 5, (0, 2, 5, 10, 15), 0, 99),
        (f"{destination}_released", tab, "Pinza: pieza soltada (%)", 20, (10, 15, 20, 25, 30, 40, 50), 1, 100),
    ])
    BOOL_FIELDS.extend([
        (f"{destination}_clearance", tab, "Compensación de altura", True),
        (f"{destination}_finish", tab, "Parar al volver a reposo", True),
    ])


def default_settings(**overrides) -> dict:
    values = {key: default for key, _, _, default, *_ in NUMERIC_FIELDS}
    values.update({key: default for key, _, _, default in BOOL_FIELDS})
    values.update(overrides)
    return values


def parse_settings(values: dict) -> dict:
    """Reject invalid edits before starting a cycle, including NaN and partial numbers."""
    parsed = default_settings()
    parsed.update(values)
    for key, _, label, default, _, minimum, maximum in NUMERIC_FIELDS:
        value = parsed[key]
        if default is None and (value is None or str(value).strip() == "Modelo"):
            parsed[key] = None
            continue
        try:
            number = float(str(value).strip().replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"{label}: escribe un número válido") from exc
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"{label}: debe estar entre {minimum:g} y {maximum:g}")
        if key.endswith("_actions"):
            if not number.is_integer():
                raise ValueError(f"{label}: debe ser un número entero")
            number = int(number)
        parsed[key] = number
    for key, _, label, _ in BOOL_FIELDS:
        value = parsed[key]
        if isinstance(value, bool):
            continue
        if value not in ("Sí", "No"):
            raise ValueError(f"{label}: elige Sí o No")
        parsed[key] = value == "Sí"
    for destination in ("green", "white"):
        if parsed[f"{destination}_released"] <= parsed[f"{destination}_closed"]:
            raise ValueError(f"Caja {'verde' if destination == 'green' else 'blanca'}: "
                             "el umbral de pieza soltada debe superar el inicio de apertura")
    return parsed


class MotionControls:
    """A compact tabbed panel; numeric dropdowns also accept typed values."""

    def __init__(self, parent, initial: dict | None = None):
        import tkinter as tk
        from tkinter import ttk

        self.initial = parse_settings(initial or {})
        self.variables, self.widgets = {}, {}
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="x", padx=8, pady=4)
        tabs, rows = {}, {}
        for title in ("Recogida", "Caja verde", "Caja blanca", "Parada automática", "Clasificación / control"):
            tabs[title] = ttk.Frame(notebook, padding=4)
            notebook.add(tabs[title], text=title)
            rows[title] = 0

        def add(key, title, label, choices, readonly=False):
            index = rows[title]
            row, column = index // 2, (index % 2) * 2
            ttk.Label(tabs[title], text=label).grid(row=row, column=column, sticky="w", padx=(4, 8), pady=4)
            variable = tk.StringVar(value=self.display(self.initial[key]))
            widget = ttk.Combobox(tabs[title], textvariable=variable, values=choices,
                                 state="readonly" if readonly else "normal", width=9, height=10)
            widget.bind("<Return>", lambda event: parent.focus_set())
            widget.grid(row=row, column=column + 1, sticky="ew", padx=(0, 12), pady=4)
            self.variables[key], self.widgets[key] = variable, (widget, "readonly" if readonly else "normal")
            rows[title] += 1

        for key, title, label, default, choices, *_ in NUMERIC_FIELDS:
            values = (("Modelo",) if default is None else ()) + tuple(str(value) for value in choices)
            add(key, title, label, values)
        for key, title, label, _ in BOOL_FIELDS:
            add(key, title, label, ("Sí", "No"), readonly=True)
        footer = ttk.Frame(parent)
        footer.pack(fill="x", padx=12, pady=(0, 4))
        ttk.Label(footer, text="Elige o escribe; Enter finaliza la edición y Espacio inicia. "
                  "Modelo = checkpoint. Paso máximo = distancia a la pose medida.",
                  wraplength=730).pack(side="left")
        self.reset_button = ttk.Button(footer, text="Restablecer", command=self.reset)
        self.reset_button.pack(side="right", padx=4)

    @staticmethod
    def display(value) -> str:
        if value is None:
            return "Modelo"
        if isinstance(value, bool):
            return "Sí" if value else "No"
        return f"{value:g}"

    def read(self) -> dict:
        return parse_settings({key: variable.get() for key, variable in self.variables.items()})

    def reset(self):
        for key, value in self.initial.items():
            self.variables[key].set(self.display(value))

    def set_enabled(self, enabled: bool):
        for widget, state in self.widgets.values():
            widget.configure(state=state if enabled else "disabled")
        self.reset_button.configure(state="normal" if enabled else "disabled")
