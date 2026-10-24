"""Demo intent classification and proposed plans; never executes robot actions."""
from __future__ import annotations

from pathlib import Path
import sys
import time

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


def model_plan(text, source, service):
    started = time.perf_counter()
    prediction = service.predict(text)
    confidence = prediction['confidence']
    intent = prediction['intent']
    issues = []
    messages = {'ask_box': 'Indica si las piezas van a la caja grande o a la blanca.',
                'unsure': 'La interpretación tiene poca confianza. Reformula la orden.',
                'not_a_cell_box': 'El modelo reconoce la caja grande y la blanca; ese destino no está disponible.',
                'no_action': 'No se ha reconocido una orden de manipulación compatible.',
                'answer': 'Indica qué quieres hacer con esas piezas.',
                'repeat': 'Repite la orden que quieres preparar.'}
    if prediction['status'] in messages:
        issues.append(messages[prediction['status']])
    if confidence < .85:
        issues.append('La confianza del modelo es inferior al 85 %. Confirma la orden reformulándola.')
    actions, descriptions = [], []
    controls = {'stop': 'STOP_PROCESS', 'cancel': 'CANCEL_ACTION'}
    if intent in controls:
        action = controls[intent]
        actions.append({'intent': action, 'label': INTENT_DESCRIPTIONS_ES[action], 'parameters': {},
                        'negated': False, 'confidence': confidence})
        descriptions.extend(CONTROL_PLANS[action])
    for rule in prediction['rules']:
        action = ('PICK_OBJECT' if intent == 'pick' and rule['box'] == 'none' else
                  'SORT_OBJECTS' if intent == 'sort' else 'MOVE_OBJECT')
        quantity = {'one': 1, 'two': 2, 'three': 3}.get(rule['count'])
        feature = {'bolt': 'con perno', 'no bolt': 'sin perno', 'any': None}[rule['feature']]
        quality = {'good': 'buena', 'bad': 'defectuosa', 'any': None}[rule['quality']]
        box = {'big': 'grande', 'white': 'blanca', 'other': None, 'none': None}[rule['box']]
        params = {'object_type': 'pieza', 'object_color': None, 'object_feature': feature,
                  'object_quality': quality, 'quantity': quantity,
                  'quantity_mode': 'todas' if rule['count'] == 'all' else None,
                  'destination_type': 'caja' if box else None, 'destination_color': None,
                  'destination_role': box}
        actions.append({'intent': action, 'label': INTENT_DESCRIPTIONS_ES[action], 'parameters': params,
                        'negated': False, 'confidence': min(confidence, rule['confidence'])})
        who = f"{quantity} {'pieza' if quantity == 1 else 'piezas'}" if quantity else 'las piezas'
        if quality:
            who += ' ' + quality + ('' if quantity == 1 else 's')
        if feature:
            who += ' ' + feature
        descriptions.extend([f'Localizar {who}', 'Comprobar alcance y espacio libre', f'Recoger {who}'])
        if box:
            descriptions.extend([f'Transportar a la caja {box}', 'Soltar las piezas',
                                 'Verificar la colocación con la cámara'])
        elif action == 'PICK_OBJECT':
            descriptions.append('Verificar la sujeción')
    valid = prediction['status'] in ('ok', 'stop', 'cancel') and bool(descriptions) and not issues
    return {'text': text.strip(), 'source': source, 'classifier': f'GLiNER2 · {service.model_name}',
            'status': 'proposed' if valid else 'needs_clarification', 'actions': actions,
            'steps': [{'number': i + 1, 'text': step, 'status': 'proposed'}
                      for i, step in enumerate(descriptions)] if valid else [],
            'issues': list(dict.fromkeys(issues)), 'execution': 'preview_only',
            'confidence': confidence, 'probabilities': prediction['probabilities'],
            'model_status': prediction['status'],
            'processing_time_ms': round((time.perf_counter() - started) * 1000, 2)}


def make_plan(text, source='text', intent_service=None):
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError('Escribe una orden de entre 1 y 2000 caracteres.')
    if source not in ('text', 'voice'):
        raise ValueError('Fuente de orden no válida.')
    command = understand_text(text.strip(), source=source).to_dict()
    # Keep the existing controls, inspection and colour/shape commands outside the
    # two-box model's taxonomy. Their response remains explicitly labelled as rules.
    rule_only = any(action['intent'] in CONTROL_PLANS
                    or action['intent'] in ('INSPECT_OBJECT', 'INSPECT_BATCH', 'COUNT_OBJECTS',
                                            'LOCATE_OBJECT', 'SHOW_OBJECTS', 'SELECT_OBJECTS')
                    or (action['intent'] != 'UNKNOWN' and
                        (action['parameters'].get('object_color')
                         or action['parameters'].get('destination_color') not in (None, 'blanco', 'blanca')
                         or action['parameters'].get('object_type') not in (None, 'pieza')))
                    for action in command['actions'])
    if intent_service is not None and not rule_only:
        return model_plan(text, source, intent_service)
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
