// F1 + glue — bootstrap na main thread: getUserMedia, loop por frame, deteccao facial
// (MediaPipe), recorte do(s) rosto(s), e envio ao worker que classifica a emocao.
//
// Divisao de trabalho:
//   main thread  : camera, deteccao facial (barata), recorte, overlay, texto do label,
//                  suavizacao temporal (F6) e mapeamento p/ o contrato Rekognition
//   Web Worker   : APENAS a inferencia de emocao (session.run) — a parte que travaria a UI
//
// Multiplos rostos: cada rosto detectado recebe um id estavel do FaceTracker (IoU
// frame-a-frame) e seu proprio estado (`faces` abaixo) — smoother, stickyLabel e busy
// independentes, para um rosto nao "roubar" o label/suavizacao de outro. Uma unica sessao
// ONNX no worker processa os recortes em fila (ver worker.js); o `id` de cada mensagem
// roteia a resposta de volta ao rosto certo, mesmo se ele ja tiver saido de cena.

import './styles.css';
import { initFaceDetector, detectFace } from './face-detect.js';
import { toRekognition } from './emotions.js';
import { EmaSmoother } from './smoothing.js';
import { FaceTracker } from './face-tracker.js';

const video = document.getElementById('cam');
const stageEl = document.getElementById('stage');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const labelEl = document.getElementById('label');
const statusEl = document.getElementById('status');
const barsEl = document.getElementById('bars');

// Limiar minimo p/ TROCAR o label exibido — evita "piscar" entre classes empatadas
// (a suavizacao temporal e no worker->main; aqui e so histerese de exibicao). Relatorio secao 4.
const SWITCH_THRESHOLD = 0.4;
const FACE_SIZE = 224; // input do enet_b0_8_best_afew (EfficientNet-B0)
const FACE_COLORS = ['#38bdf8', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#fb923c'];

const tracker = new FaceTracker();
/** @type {Map<number, { box: object, smoother: EmaSmoother, stickyLabel: string|null, busy: boolean, emotions: object[] }>} */
const faces = new Map();
const chipEls = new Map(); // id -> elemento .face-chip

let worker = null;
let faceDelegate = null;
let lastTs = 0;
let lastVisible = []; // ultimo [{id, box}] visto — resultados assincronos do worker redesenham a partir dele

function setStatus(text, kind = 'info') {
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
}

function colorForId(id) {
  return FACE_COLORS[(id - 1) % FACE_COLORS.length];
}

async function boot() {
  if (!self.crossOriginIsolated) {
    console.warn('[fer] crossOriginIsolated=false — threads WASM desabilitadas (single-thread)');
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('Sem acesso à câmera: contexto inseguro. Abra por https:// ou http://localhost.', 'error');
    return;
  }

  if ('serviceWorker' in navigator) {
    // F6 — cache-first para /models, /mediapipe e /assets: nao re-baixa a cada visita.
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  }

  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => setStatus(`Falha no worker: ${e.message}`, 'error');

  setStatus('Pedindo acesso à câmera…');
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch {
    setStatus('Permissão de câmera negada. Libere o acesso nas configurações do site e recarregue.', 'error');
    return;
  }

  video.srcObject = stream;
  await video.play();
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  setStatus('Carregando modelos (detecção + emoção)…');
  try {
    faceDelegate = await initFaceDetector();
  } catch (err) {
    setStatus(`Erro ao carregar o detector facial: ${err?.message || err}`, 'error');
    return;
  }
  worker.postMessage({ type: 'init' });
}

function onWorkerMessage(ev) {
  const msg = ev.data;
  switch (msg.type) {
    case 'ready':
      setStatus(`Pronto — rosto em ${faceDelegate}, emoção em ${msg.ep.toUpperCase()}.`, 'ok');
      scheduleNextFrame();
      break;
    case 'init-error':
      setStatus(`Erro ao iniciar a inferência: ${msg.error}`, 'error');
      break;
    case 'result': {
      const face = faces.get(msg.id);
      if (!face) break; // rosto ja saiu de cena antes da resposta chegar — descarta
      face.busy = false;
      if (msg.error) console.warn('[fer] frame:', msg.id, msg.error);
      if (msg.probs) {
        face.emotions = toRekognition(face.smoother.push(msg.probs));
        const top = face.emotions[0];
        if (top && (top.confidence >= SWITCH_THRESHOLD || face.stickyLabel === null)) {
          face.stickyLabel = top.type;
        }
      }
      renderChips(lastVisible);
      renderBars(lastVisible);
      break;
    }
    default:
      break;
  }
}

function scheduleNextFrame() {
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(onFrame);
  } else {
    // Fallback: rAF. Nao ideal (relatorio 2.6), mas mantem o loop vivo.
    requestAnimationFrame(() => onFrame(performance.now()));
  }
}

function onFrame(now) {
  if (video.readyState >= 2) {
    // Timestamp estritamente crescente exigido pelo modo VIDEO do MediaPipe.
    const ts = Math.max(lastTs + 1, Math.round(now || performance.now()));
    lastTs = ts;

    let boxes = [];
    try {
      boxes = detectFace(video, ts, video.videoWidth, video.videoHeight);
    } catch (err) {
      console.warn('[fer] detecção:', err);
    }

    const { visible, aliveIds } = tracker.update(boxes);
    lastVisible = visible;
    syncFaces(aliveIds, visible);

    drawOverlay(visible);
    renderChips(visible);
    renderBars(visible);
    setLabelCount(visible.length);

    for (const { id, box } of visible) {
      const face = faces.get(id);
      if (face.busy) continue;
      face.busy = true;
      sendFaceFrame(id, box);
    }
  }
  scheduleNextFrame();
}

function syncFaces(aliveIds, visible) {
  for (const id of faces.keys()) {
    if (!aliveIds.has(id)) faces.delete(id);
  }
  for (const { id, box } of visible) {
    const existing = faces.get(id);
    if (existing) {
      existing.box = box;
    } else {
      faces.set(id, { box, smoother: new EmaSmoother(0.6), stickyLabel: null, busy: false, emotions: [] });
    }
  }
}

async function sendFaceFrame(id, box) {
  try {
    const bitmap = await createImageBitmap(video, box.x, box.y, box.w, box.h, {
      resizeWidth: FACE_SIZE,
      resizeHeight: FACE_SIZE,
      resizeQuality: 'medium',
    });
    worker.postMessage({ type: 'frame', id, bitmap }, [bitmap]);
  } catch {
    const face = faces.get(id);
    if (face) face.busy = false; // descarta este frame
  }
}

function drawOverlay(visible) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  for (const { id, box } of visible) {
    octx.strokeStyle = colorForId(id);
    octx.lineWidth = 3;
    octx.strokeRect(box.x, box.y, box.w, box.h);
  }
}

function setLabelCount(n) {
  if (n === 0) {
    labelEl.textContent = 'nenhum rosto';
    labelEl.dataset.kind = 'none';
  } else {
    labelEl.textContent = n === 1 ? '1 rosto' : `${n} rostos`;
    labelEl.dataset.kind = 'face';
  }
}

// Chips flutuantes por rosto, ancorados acima de cada caixa. Ficam FORA do <canvas>
// espelhado (CSS transform: scaleX(-1) em #cam/#overlay) — por isso a posicao e calculada
// aqui em % ja espelhada, mas o texto em si nao precisa de nenhum flip manual.
function renderChips(visible) {
  const seen = new Set();
  for (const { id, box } of visible) {
    seen.add(id);
    const face = faces.get(id);
    let chip = chipEls.get(id);
    if (!chip) {
      chip = document.createElement('div');
      chip.className = 'face-chip';
      stageEl.appendChild(chip);
      chipEls.set(id, chip);
    }
    chip.style.setProperty('--chip-color', colorForId(id));
    const leftPct = 100 - ((box.x + box.w) / video.videoWidth) * 100;
    const topPct = (box.y / video.videoHeight) * 100;
    chip.style.left = `${leftPct}%`;
    chip.style.top = `${topPct}%`;
    const top = face?.emotions?.[0];
    chip.textContent = face?.stickyLabel
      ? `${face.stickyLabel} · ${Math.round((top?.confidence ?? 0) * 100)}%`
      : '…';
  }
  for (const [id, chip] of chipEls) {
    if (!seen.has(id)) {
      chip.remove();
      chipEls.delete(id);
    }
  }
}

function buildBarRow(e) {
  const row = document.createElement('div');
  row.className = 'bar';

  const name = document.createElement('span');
  name.className = 'bar-name';
  name.textContent = e.type;

  const track = document.createElement('span');
  track.className = 'bar-track';
  const fill = document.createElement('span');
  fill.className = 'bar-fill';
  fill.style.width = `${(e.confidence * 100).toFixed(1)}%`;
  track.append(fill);

  const val = document.createElement('span');
  val.className = 'bar-val';
  val.textContent = Math.round(e.confidence * 100);

  row.append(name, track, val);
  return row;
}

function renderBars(visible) {
  if (visible.length === 0) {
    barsEl.replaceChildren();
    return;
  }
  const groups = visible.map(({ id }) => {
    const face = faces.get(id);
    const group = document.createElement('div');
    group.className = 'face-group';
    if (visible.length > 1) {
      const heading = document.createElement('div');
      heading.className = 'face-group-heading';
      heading.style.setProperty('--chip-color', colorForId(id));
      heading.textContent = `Rosto ${id}`;
      group.append(heading);
    }
    group.append(...(face?.emotions ?? []).map(buildBarRow));
    return group;
  });
  barsEl.replaceChildren(...groups);
}

boot();
