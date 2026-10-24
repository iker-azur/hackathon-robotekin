// Local classification controls; voice input is handled by the Telegram bot.
export function initControlInterface(root = document, browser = window) {
  const $ = id => root.getElementById(id);
  let ruleSignature = null;
  let busy = false;
  let robotState = null;

  function complementary(value) {
    return value === 'with_pin' || value === 'good' ?
      (value === 'with_pin' ? 'without_pin' : 'bad') :
      (value === 'without_pin' ? 'with_pin' : 'good');
  }

  function setRule(ruleState) {
    const signature = JSON.stringify(ruleState);
    if (signature === ruleSignature) return;
    ruleSignature = signature;
    const rule = ruleState?.rule;
    if (rule) {
      $('good-condition').value = rule.good_has_pin ? 'with_pin' : 'without_pin';
      $('bad-condition').value = complementary($('good-condition').value);
      $('green-quality').value = rule.good_destination === 'green' ? 'good' : 'bad';
      $('white-quality').value = complementary($('green-quality').value);
      $('rule-status').textContent = 'Configuración activa';
    } else {
      $('good-condition').value = 'with_pin';
      $('bad-condition').value = 'without_pin';
      $('green-quality').value = 'good';
      $('white-quality').value = 'bad';
      $('rule-status').textContent = 'Configuración sin aplicar';
    }
  }

  function markEdited() {
    $('rule-status').textContent = 'Cambios sin aplicar';
  }

  async function applyRule() {
    if (busy) return;
    const good = $('good-condition').value;
    const bad = $('bad-condition').value;
    const green = $('green-quality').value;
    const white = $('white-quality').value;
    if (good === bad || green === white) {
      $('control-feedback').textContent = 'Las piezas y las cajas deben tener asignaciones opuestas.';
      return;
    }
    const condition = value => value === 'with_pin' ? 'con perno' : 'sin perno';
    const destination = quality => green === quality ? 'verde' : 'blanca';
    const text = `Las piezas ${condition(good)} son buenas y van a la caja ${destination('good')}. ` +
      `Las piezas ${condition(bad)} son malas y van a la caja ${destination('bad')}.`;
    busy = true;
    $('apply-rule').disabled = true;
    $('control-feedback').textContent = 'Aplicando configuración…';
    try {
      const response = await browser.fetch('/api/sorting-rule', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `Error ${response.status}`);
      if (result.status !== 'active') throw new Error(result.message || 'No se pudo interpretar la configuración.');
      ruleSignature = null;
      setRule({ active: true, rule: result.rule });
      $('control-feedback').textContent = 'Configuración aplicada a la próxima recogida.';
      root.dispatchEvent(new browser.CustomEvent('hmi:sorting-rule', { detail: { active: true, rule: result.rule } }));
    } catch (error) {
      $('control-feedback').textContent = `No se pudo aplicar: ${error.message}`;
    } finally {
      busy = false;
      $('apply-rule').disabled = false;
    }
  }

  function updateRobot(state) {
    robotState = state;
    const labels = {
      starting: 'Preparando aplicación', idle: 'En reposo', running: 'Recogida en curso',
      classifying: 'Clasificando pieza', routing: 'Trasladando a la caja',
      result: 'Ciclo terminado', manual: 'Modo manual activo', homing: 'Volviendo a casa', offline: 'Aplicación desconectada',
    };
    const connected = Boolean(state?.connected);
    const phase = connected ? state.phase : 'offline';
    $('robot-app-state').textContent = connected ? 'APLICACIÓN CONECTADA' : 'APLICACIÓN DESCONECTADA';
    $('robot-phase').textContent = labels[phase] || 'Estado desconocido';
    $('robot-plan-detail').textContent = connected
      ? (state.note || (phase === 'idle' ? 'Lista para iniciar un ciclo.' : 'Planificación en curso.'))
      : 'Inicia la aplicación de recogida para ver el plan real.';
    $('robot-plan-result').textContent = state?.result || '';
    const destinations = { green: 'caja verde', white: 'caja blanca' };
    $('robot-plan-destination').textContent = state?.destination
      ? `Destino actual: ${destinations[state.destination]}` : '';
    $('robot-plan-settings').textContent = state?.settings?.pick_seconds
      ? `Duración máxima de recogida: ${state.settings.pick_seconds} s` : '';
    $('start-cycle').disabled = busy || !connected || !['idle', 'result'].includes(phase);
    $('home-cycle').disabled = busy || !connected || !['idle', 'result'].includes(phase) ||
      state?.control_mode !== 'automatic';
    if (root.activeElement !== $('robot-control-mode')) {
      $('robot-control-mode').value = state?.control_mode === 'manual' ? 'manual' : 'automatic';
    }
    $('robot-control-mode').disabled = !connected || !['idle', 'result', 'manual'].includes(phase);
    $('apply-control-mode').disabled = busy || !connected || !['idle', 'result', 'manual'].includes(phase) ||
      ($('robot-control-mode').value === 'manual' && !state.leader_available) ||
      $('robot-control-mode').value === (state?.control_mode === 'manual' ? 'manual' : 'automatic');
    $('control-mode-status').textContent = !connected ? 'Conecta la aplicación para cambiar de modo.'
      : !state.leader_available ? (state.leader_status || 'Maestro no conectado o sin calibración disponible.')
        : state.control_mode === 'manual' ? 'Modo manual activo: el follower sigue al maestro.'
          : 'Modo automático activo: el programa controla el follower.';
    const selectedMode = $('robot-control-mode').value;
    const currentMode = state?.control_mode === 'manual' ? 'manual' : 'automatic';
    $('control-mode-status').textContent = !connected ? 'Conecta la aplicación para cambiar de modo.'
      : selectedMode === 'manual' && !state.leader_available
        ? (state.leader_status || 'Maestro no conectado o sin calibración disponible.')
        : selectedMode !== currentMode
          ? 'Leader verificado. Pulsa CAMBIAR MODO para activar el modo seleccionado.'
          : currentMode === 'manual' ? 'Modo manual activo: el follower sigue al maestro.'
            : 'Modo automático activo: el programa controla al follower.';
    if (!state?.demo) {
      root.querySelectorAll('.inspection-flow li').forEach(item => item.classList.remove('active'));
    }
    if (phase === 'running') root.querySelector('[data-stage="recogiendo"]').classList.add('active');
    if (phase === 'classifying') root.querySelector('[data-stage="clasificando"]').classList.add('active');
    if (phase === 'routing') root.querySelector('[data-stage="depositando"]').classList.add('active');
  }

  async function startCycle() {
    if (busy || $('start-cycle').disabled) return;
    busy = true;
    $('start-cycle').disabled = true;
    $('control-feedback').textContent = 'Solicitando un ciclo…';
    try {
      const response = await browser.fetch('/api/robot/start', {
        method: 'POST',
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'La aplicación no está disponible.');
      $('control-feedback').textContent = 'Ciclo solicitado a la aplicación.';
    } catch (error) {
      $('control-feedback').textContent = error.message;
    } finally {
      busy = false;
      const phase = $('robot-app-state').textContent;
      $('start-cycle').disabled = phase !== 'APLICACIÓN CONECTADA' ||
        !['En reposo', 'Ciclo terminado'].includes($('robot-phase').textContent);
    }
  }

  async function applyControlMode() {
    if (busy || $('apply-control-mode').disabled) return;
    const mode = $('robot-control-mode').value;
    if (mode === 'manual' && !browser.confirm('El follower seguirá los movimientos del maestro. ¿Activar el modo manual?')) return;
    busy = true;
    $('apply-control-mode').disabled = true;
    $('control-feedback').textContent = 'Solicitando cambio de control…';
    try {
      const response = await browser.fetch('/api/robot/control', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `Error ${response.status}`);
      $('control-feedback').textContent = result.detail;
    } catch (error) {
      $('control-feedback').textContent = error.message;
    } finally {
      busy = false;
      if (robotState) updateRobot(robotState);
    }
  }

  async function requestHome() {
    if (busy || $('home-cycle').disabled) return;
    if (!browser.confirm('El follower se moverá a la posición de casa guardada. ¿Continuar?')) return;
    busy = true;
    $('home-cycle').disabled = true;
    $('control-feedback').textContent = 'Solicitando vuelta a casa…';
    try {
      const response = await browser.fetch('/api/robot/home', { method: 'POST' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `Error ${response.status}`);
      $('control-feedback').textContent = result.detail;
    } catch (error) {
      $('control-feedback').textContent = error.message;
    } finally {
      busy = false;
      if (robotState) updateRobot(robotState);
    }
  }

  $('good-condition').addEventListener('change', () => {
    $('bad-condition').value = complementary($('good-condition').value);
    markEdited();
  });
  $('bad-condition').addEventListener('change', () => {
    $('good-condition').value = complementary($('bad-condition').value);
    markEdited();
  });
  $('green-quality').addEventListener('change', () => {
    $('white-quality').value = complementary($('green-quality').value);
    markEdited();
  });
  $('white-quality').addEventListener('change', () => {
    $('green-quality').value = complementary($('white-quality').value);
    markEdited();
  });
  $('apply-rule').addEventListener('click', applyRule);
  $('home-cycle').addEventListener('click', requestHome);
  $('start-cycle').addEventListener('click', startCycle);
  $('apply-control-mode').addEventListener('click', applyControlMode);
  $('robot-control-mode').addEventListener('change', () => {
    if (robotState) updateRobot(robotState);
  });
  root.addEventListener('hmi:sorting-rule', event => setRule(event.detail));
  root.addEventListener('hmi:robot-state', event => updateRobot(event.detail));

  browser.fetch('/api/sorting-rule').then(response => response.json()).then(setRule).catch(() => {});
  return { setRule, updateRobot };
}

if (typeof document !== 'undefined') initControlInterface();
