// Microphone capture is independent of the 3D renderer and never controls motors.
export function initVoiceInterface(root = document, browser = window) {
  const $ = id => root.getElementById(id);
  let stream = null;
  let recorder = null;
  let audioContext = null;
  let meterFrame = null;
  let recordingTimer = null;
  let limitTimer = null;
  let audioURL = null;
  let busy = false;
  let recording = false;
  let disposed = false;
  let cancelled = false;
  let source = 'text';
  let hasPlan = false;
  let requestController = null;
  function notify(type, detail) {
    if (root.dispatchEvent && browser.CustomEvent) root.dispatchEvent(new browser.CustomEvent(type, { detail }));
  }

  function status(label, detail, active = false) {
    $('voice-status').textContent = label;
    $('voice-help').textContent = detail;
    $('voice-status').classList.toggle('recording', active);
    $('microphone').classList.toggle('recording', active);
    $('microphone').setAttribute('aria-pressed', String(active));
    if (active) notify('hmi:phase', 'listening');
    else if (label === 'TRANSCRIBIENDO') notify('hmi:phase', 'understanding');
    else if (['ERROR', 'AUDIO GUARDADO', 'SIN AUDIO', 'MICRÓFONO NO DISPONIBLE', 'CAPTURA NO DISPONIBLE', 'ERROR DE CAPTURA'].includes(label)) notify('hmi:phase', 'idle');
  }
  function setBusy(value) {
    busy = value;
    $('microphone').disabled = value;
    $('analyze-command').disabled = value || recording;
    $('command-text').readOnly = value || recording;
    root.querySelectorAll('[data-command]').forEach(button => { button.disabled = value || recording; });
  }
  function clearPlan(label = 'SIN ANALIZAR') {
    hasPlan = false;
    $('plan-badge').textContent = label;
    $('plan-intents').replaceChildren();
    $('plan-steps').replaceChildren();
    $('plan-issues').replaceChildren();
    $('plan-issues').hidden = true;
    const text = root.createElement('p');
    text.textContent = 'Analiza la orden para actualizar la intención y el plan.';
    $('plan-summary').replaceChildren(text);
    if ($('plan-panel')) $('plan-panel').hidden = label !== 'ANALIZANDO';
    notify('hmi:plan', null);
    notify('hmi:phase', label === 'ANALIZANDO' ? 'understanding' : label === 'ESCUCHANDO' ? 'listening' : 'idle');
  }
  function renderPlan(plan) {
    hasPlan = true;
    $('plan-badge').textContent = plan.status === 'proposed' ? 'PLAN PROPUESTO' : 'NECESITA ACLARACIÓN';
    const title = root.createElement('h2');
    title.textContent = plan.actions.map(action => action.label).join(' → ') || 'Intención no reconocida';
    const meta = root.createElement('p');
    meta.textContent = `${plan.classifier} · ${plan.source === 'voice' ? 'Entrada por voz' : 'Entrada de texto'}`;
    $('plan-summary').replaceChildren(title, meta);
    $('plan-intents').replaceChildren();
    for (const action of plan.actions) {
      const tag = root.createElement('span');
      tag.className = 'intent-tag';
      tag.textContent = `${action.negated ? 'NO · ' : ''}${action.intent}`;
      $('plan-intents').append(tag);
      const labels = { object_type: 'Objeto', object_color: 'Color', quantity: 'Cantidad', object_quality: 'Calidad', destination_type: 'Destino', destination_color: 'Color destino', destination_role: 'Zona' };
      const params = root.createElement('span');
      params.className = 'intent-parameters';
      params.textContent = Object.entries(action.parameters).filter(([, value]) => value !== null && value !== 'UNSPECIFIED' && value !== '').map(([key, value]) => `${labels[key] || key}: ${value}`).join(' · ');
      if (params.textContent) $('plan-intents').append(params);
    }
    $('plan-issues').replaceChildren();
    $('plan-issues').hidden = plan.issues.length === 0;
    for (const issue of plan.issues) {
      const line = root.createElement('p');
      line.textContent = issue;
      $('plan-issues').append(line);
    }
    $('plan-steps').replaceChildren();
    for (const step of plan.steps) {
      const row = root.createElement('li');
      const number = root.createElement('span');
      number.className = 'plan-number';
      number.textContent = String(step.number).padStart(2, '0');
      const text = root.createElement('span');
      text.textContent = step.text;
      row.append(number, text);
      $('plan-steps').append(row);
    }
    if ($('plan-panel')) $('plan-panel').hidden = false;
    notify('hmi:plan', plan);
  }
  async function jsonRequest(url, options = {}) {
    const response = await browser.fetch(url, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `Error ${response.status}`);
    return result;
  }
  async function analyze() {
    const text = $('command-text').value.trim();
    if (!text) {
      clearPlan('SIN ORDEN');
      status('FALTA UNA ORDEN', 'Graba una orden o escribe el texto que quieres analizar.');
      return;
    }
    setBusy(true);
    clearPlan('ANALIZANDO');
    try {
      const plan = await jsonRequest('/api/command', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source }),
      });
      if (disposed) return;
      renderPlan(plan);
      status('MICRÓFONO APAGADO', plan.status === 'proposed'
        ? 'Intención y plan preparados. Puedes corregir la orden o grabar otra.'
        : 'La orden necesita aclaración. Revisa el mensaje del panel de planificación.');
    } catch (error) {
      if (disposed) return;
      clearPlan('ERROR DE ANÁLISIS');
      status('ERROR', `No se pudo analizar la orden: ${error.message}`);
    } finally {
      if (!disposed) setBusy(false);
    }
  }
  function releaseMicrophone() {
    stream?.getTracks().forEach(track => track.stop());
    stream = null;
    browser.clearInterval(recordingTimer);
    browser.clearTimeout(limitTimer);
    if (meterFrame !== null) browser.cancelAnimationFrame(meterFrame);
    meterFrame = null;
    if (audioContext) { audioContext.close().catch(() => {}); audioContext = null; }
    $('audio-level-fill').style.width = '0%';
    $('audio-level').setAttribute('aria-valuenow', '0');
    notify('hmi:level', 0);
  }
  function startMeter(activeStream) {
    const Context = browser.AudioContext || browser.webkitAudioContext;
    if (!Context) return;
    try {
      audioContext = new Context();
      audioContext.resume().catch(() => {});
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      audioContext.createMediaStreamSource(activeStream).connect(analyser);
      const samples = new Uint8Array(analyser.fftSize);
      function meter() {
        if (!recording) return;
        analyser.getByteTimeDomainData(samples);
        const rms = Math.sqrt(samples.reduce((sum, value) => sum + ((value - 128) / 128) ** 2, 0) / samples.length);
        const level = Math.min(100, Math.round(rms * 400));
        notify('hmi:level', level / 100);
        $('audio-level-fill').style.width = `${level}%`;
        $('audio-level').setAttribute('aria-valuenow', String(level));
        meterFrame = browser.requestAnimationFrame(meter);
      }
      meter();
    } catch (error) { /* Recording still works if only the level meter is unavailable. */ }
  }
  async function finishRecording(chunks, mimeType) {
    recording = false;
    releaseMicrophone();
    $('microphone-label').textContent = 'Activar micrófono';
    if (cancelled || disposed) { if (!disposed) setBusy(false); return; }
    setBusy(true);
    status('AUDIO CAPTURADO', 'Grabación finalizada. Preparando la transcripción local…');
    const blob = new browser.Blob(chunks, { type: mimeType });
    if (!blob.size) {
      status('SIN AUDIO', 'La grabación no contiene audio. Vuelve a intentarlo.');
      setBusy(false);
      return;
    }
    if (audioURL) browser.URL.revokeObjectURL(audioURL);
    audioURL = browser.URL.createObjectURL(blob);
    $('audio-playback').src = audioURL;
    $('audio-download').href = audioURL;
    $('audio-download').download = `orden.${mimeType.includes('ogg') ? 'ogg' : mimeType.includes('mp4') ? 'm4a' : 'webm'}`;
    $('recorded-audio').hidden = false;
    requestController = new browser.AbortController();
    const timeout = browser.setTimeout(() => requestController?.abort(), 180000);
    try {
      const capability = await jsonRequest('/api/voice/status');
      if (!capability.available) {
        $('voice-setup').hidden = false;
        $('voice-install').textContent = capability.install_command;
        throw new Error('Falta instalar Whisper. Puedes escuchar el audio y escribir la orden mientras preparas la transcripción.');
      }
      status('TRANSCRIBIENDO', 'Whisper está procesando el audio. La primera vez puede descargar y cargar el modelo.');
      const result = await jsonRequest('/api/voice/transcribe', {
        method: 'POST', headers: { 'Content-Type': blob.type }, body: blob,
        signal: requestController.signal,
      });
      if (disposed) return;
      $('command-text').value = result.text;
      source = 'voice';
      await analyze();
    } catch (error) {
      if (!disposed) status('AUDIO GUARDADO', error.name === 'AbortError'
        ? 'La transcripción ha tardado demasiado. El audio sigue disponible; puedes escribir la orden.'
        : error.message);
    } finally {
      browser.clearTimeout(timeout);
      requestController = null;
      if (!disposed) setBusy(false);
    }
  }
  function stopRecording() {
    if (recorder && recorder.state !== 'inactive') {
      $('microphone').disabled = true;
      recorder.stop();
      releaseMicrophone();
    }
  }
  async function startRecording() {
    if (busy || recording || disposed) return;
    if (!browser.navigator.mediaDevices?.getUserMedia || !browser.MediaRecorder) {
      status('CAPTURA NO DISPONIBLE', 'Este navegador no permite grabar audio aquí. Abre la HMI en localhost y utiliza Firefox o Chrome; también puedes escribir la orden.');
      return;
    }
    setBusy(true);
    status('ACTIVANDO MICRÓFONO', 'Permite el acceso al micrófono en el navegador.');
    try {
      stream = await browser.navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: false });
      if (disposed) { releaseMicrophone(); return; }
      cancelled = false;
      const mimeType = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4'].find(type => browser.MediaRecorder.isTypeSupported(type));
      recorder = new browser.MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      const chunks = [];
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => finishRecording(chunks, recorder.mimeType);
      recorder.onerror = () => {
        cancelled = true;
        recording = false;
        releaseMicrophone();
        $('microphone-label').textContent = 'Activar micrófono';
        status('ERROR DE CAPTURA', 'El navegador no pudo completar la grabación. Vuelve a intentarlo.');
        setBusy(false);
      };
      stream.getAudioTracks().forEach(track => track.addEventListener('ended', stopRecording, { once: true }));
      clearPlan('ESCUCHANDO');
      $('command-text').value = '';
      source = 'text';
      $('recorded-audio').hidden = true;
      $('audio-playback').pause();
      recorder.start(250);
      recording = true;
      setBusy(false);
      $('microphone-label').textContent = 'Detener y transcribir';
      status('ESCUCHANDO', 'Di tu orden y pulsa de nuevo para terminar. La grabación se detendrá a los 30 segundos.', true);
      startMeter(stream);
      const started = Date.now();
      $('record-time').textContent = '00:00';
      recordingTimer = browser.setInterval(() => {
        $('record-time').textContent = `00:${String(Math.min(30, Math.floor((Date.now() - started) / 1000))).padStart(2, '0')}`;
      }, 200);
      limitTimer = browser.setTimeout(stopRecording, 30000);
    } catch (error) {
      recording = false;
      releaseMicrophone();
      const messages = {
        NotAllowedError: 'Acceso al micrófono denegado. Permítelo desde el icono de permisos del navegador y vuelve a pulsar.',
        NotFoundError: 'No se ha encontrado un micrófono. Conecta uno o escribe la orden.',
        NotReadableError: 'No se puede utilizar el micrófono. Comprueba si otra aplicación lo está usando.',
      };
      status('MICRÓFONO NO DISPONIBLE', messages[error.name] || `No se pudo iniciar la grabación: ${error.message}`);
      setBusy(false);
    }
  }
  $('microphone').addEventListener('click', () => recording ? stopRecording() : startRecording());
  $('command-form').addEventListener('submit', event => { event.preventDefault(); if (!busy && !recording) analyze(); });
  $('command-text').addEventListener('input', () => { source = 'text'; if (hasPlan) clearPlan(); });
  root.querySelectorAll('[data-command]').forEach(button => button.addEventListener('click', () => {
    if (busy || recording) return;
    $('command-text').value = button.dataset.command;
    source = 'text';
    analyze();
  }));
  function destroy() {
    disposed = true;
    cancelled = true;
    stopRecording();
    releaseMicrophone();
    requestController?.abort();
    if (audioURL) browser.URL.revokeObjectURL(audioURL);
  }
  browser.addEventListener('pagehide', destroy);
  async function checkVoice() {
    try {
      const capability = await jsonRequest('/api/voice/status');
      if (disposed) return;
      $('voice-setup').hidden = capability.available;
      $('voice-install').textContent = capability.install_command;
      if (!capability.available && !busy && !recording) $('voice-help').textContent = 'Whisper sin preparar · Puedes escribir la orden.';
    } catch (error) { if (!busy && !recording) $('voice-help').textContent = 'Sin conexión.'; }
  }
  checkVoice();
  return { startRecording, stopRecording, analyze, destroy };
}

if (typeof document !== 'undefined') initVoiceInterface();
