"""Demo intent classification and proposed plans; never executes robot actions."""
from __future__ import annotations

from pathlib import Path
import sys

ROBOQC = Path(__file__).resolve().parents[1] / 'roboqc' / 'phase_1' / 'src'
if str(ROBOQC) not in sys.path:
    sys.path.append(str(ROBOQC))
from roboqc.commands.pipeline import understand_text
from roboqc.commands.taxonomy import INTENT_DESCRIPTIONS_ES


CONTROL_PLANS = {
    'START_PROCESS': ['Comprobar el estado de la célula', 'Preparar el inicio del proceso'],
    'STOP_PROCESS': ['Solicitar la detención del proceso', 'Comprobar que el movimiento se ha detenido'],
    'PAUSE_PROCESS': ['Guardar el paso actual', 'Solicitar la pausa del proceso'],
    'RESUME_PROCESS': ['Comprobar el estado del proceso', 'Retomar desde el paso pendiente'],
    'CANCEL_ACTION': ['Descartar la acción pendiente', 'Actualizar el plan de trabajo'],
    'REQUEST_STATUS': ['Consultar el estado de la célula', 'Mostrar el resumen de actividad'],
    'REQUEST_REPORT': ['Recopilar los resultados disponibles', 'Mostrar el informe de inspección'],
    'EMERGENCY_STOP': ['Solicitar la parada de emergencia', 'Comprobar el estado de la célula'],
}


def target_text(parameters):
    quantity = parameters.get('quantity')
    prefix = str(quantity) + ' ' if isinstance(quantity, int) else ''
    kind = parameters.get('object_type') or 'pieza'
    if isinstance(quantity, int) and quantity > 1 and not kind.endswith('s'):
        kind += 's'
    color = parameters.get('object_color')
    quality = parameters.get('object_quality')
    return ' '.join(str(part) for part in [prefix + kind, 'de color ' + color if color else None, quality] if part)


def make_plan(text, source='text'):
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError('Escribe una orden de entre 1 y 2000 caracteres.')
    if source not in ('text', 'voice'):
        raise ValueError('Fuente de orden no válida.')
    command = understand_text(text.strip(), source=source).to_dict()
    issues = list(command['validation_errors']) + list(command['unresolved_references'])
    steps = []
    actions = []
    last_object = None
    pin_sort_added = False
    for action in command['actions']:
        intent = action['intent']
        params = dict(action['parameters'])
        if intent in ('PLACE_OBJECT', 'MOVE_OBJECT') and not params.get('object_type') and last_object:
            for key in ('object_type', 'object_color', 'object_quality', 'quantity'):
                params[key] = last_object.get(key)
        actions.append({**action, 'parameters': params, 'label': INTENT_DESCRIPTIONS_ES.get(intent, intent)})
        if action['negated']:
            issues.append('La orden contiene una negación. Reformúlala indicando qué quieres hacer.')
            continue
        if params.get('object_type'):
            last_object = params
        target = target_text(params)
        destination = ' '.join(str(part) for part in [params.get('destination_type'), params.get('destination_color'), params.get('destination_role')] if part)
        if intent in CONTROL_PLANS:
            descriptions = CONTROL_PLANS[intent]
        elif intent in ('INSPECT_OBJECT', 'INSPECT_BATCH', 'CLASSIFY_OBJECT', 'SORT_OBJECTS'):
            if intent in ('CLASSIFY_OBJECT', 'SORT_OBJECTS') and 'perno' in text.casefold():
                if pin_sort_added:
                    continue
                pin_sort_added = True
                descriptions = [
                    'Recoger de la cesta roja',
                    'Elevar bajo la cámara cenital',
                    'Mostrar la cara plana',
                    'Clasificar perno / sin perno',
                    'Depositar en la caja asignada',
                ]
            else:
                descriptions = [f'Observar {target}', 'Elevar la pieza bajo la cámara cenital',
                                'Mostrar la cara plana', 'Clasificar perno / sin perno',
                                'Depositar en la caja correspondiente']
        elif intent in ('PICK_OBJECT', 'PLACE_OBJECT', 'MOVE_OBJECT'):
            if not params.get('object_type') and not params.get('object_color'):
                issues.append('Indica qué pieza quieres manipular.')
            if intent in ('PLACE_OBJECT', 'MOVE_OBJECT') and not destination:
                issues.append('Indica el destino de las piezas.')
            descriptions = [f'Localizar {target}', 'Comprobar alcance y espacio libre', f'Recoger {target}']
            if intent in ('PLACE_OBJECT', 'MOVE_OBJECT'):
                descriptions += [f'Transportar a {destination or "destino pendiente"}', 'Soltar las piezas', 'Verificar la colocación con la cámara']
            else:
                descriptions += ['Verificar la sujeción']
        elif intent == 'UNKNOWN':
            issues.append('No se ha reconocido una intención compatible. Prueba con inspeccionar, colocar o pausar.')
            continue
        else:
            issues.append(f'La intención «{INTENT_DESCRIPTIONS_ES.get(intent, intent)}» todavía no tiene un plan en esta demo.')
            continue
        for description in descriptions:
            steps.append({'number': len(steps) + 1, 'text': description, 'status': 'proposed'})
    if command['status'] != 'VALID' and not issues:
        issues.append('La orden necesita aclaración antes de preparar un plan.')
    valid = bool(steps) and not issues and command['status'] == 'VALID'
    return {'text': text.strip(), 'source': source, 'classifier': 'Reglas de RoboQC · Demo',
            'status': 'proposed' if valid else 'needs_clarification', 'actions': actions,
            'steps': steps if valid else [], 'issues': list(dict.fromkeys(issues)),
            'execution': 'preview_only', 'processing_time_ms': command['processing_time_ms']}
