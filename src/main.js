// F1 + glue — bootstrap na main thread: getUserMedia, loop por frame, deteccao facial
// (MediaPipe), recorte do rosto, e envio ao worker que classifica a emocao.
//
// Divisao de trabalho:
//   main thread  : camera, deteccao facial (barata), recorte, overlay, texto do label,
//                  suavizacao temporal (F6) e mapeamento p/ o contrato Rekognition
//   Web Worker   : APENAS a inferencia de emocao (session.run) — a parte que travaria a UI

import './styles.css';
import { initFaceDetector, detectFace } from './face-detect.js';
import { toRekognition } from './emotions.js';
import { EmaSmoother } from './smoothing.js';

const video = document.getElementById('cam');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const labelEl = document.getElementById('label');
const statusEl = document.getElementById('status');
const barsEl = document.getElementById('bars');

// Limiar minimo p/ TROCAR o label exibido — evita "piscar" entre classes empatadas
// (a suavizacao temporal e no worker->main; aqui e so histerese de exibicao). Relatorio secao 4.
const SWITCH_THRESHOLD = 0.4;
const FACE_SIZE = 224; // input do enet_b0_8_best_afew (EfficientNet-B0)

const smoother = new EmaSmoother(0.6);

let worker = null;
let busy = false; // um frame no worker por vez, sem fila
let stickyLabel = null;
let faceDelegate = null;
let lastTs = 0;

function setStatus(text, kind = 'info') {
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
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
    case 'result':
      busy = false;
      if (msg.error) console.warn('[fer] frame:', msg.error);
      if (msg.probs) updateEmotions(toRekognition(smoother.push(msg.probs)));
      break;
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

async function onFrame(now) {
  if (video.readyState >= 2) {
    // Timestamp estritamente crescente exigido pelo modo VIDEO do MediaPipe.
    const ts = Math.max(lastTs + 1, Math.round(now || performance.now()));
    lastTs = ts;

    let box = null;
    try {
      box = detectFace(video, ts, video.videoWidth, video.videoHeight);
    } catch (err) {
      console.warn('[fer] detecção:', err);
    }

    drawOverlay(box);

    if (!box) {
      showNoFace();
    } else if (!busy) {
      busy = true;
      try {
        const bitmap = await createImageBitmap(video, box.x, box.y, box.w, box.h, {
          resizeWidth: FACE_SIZE,
          resizeHeight: FACE_SIZE,
          resizeQuality: 'medium',
        });
        worker.postMessage({ type: 'frame', bitmap }, [bitmap]);
      } catch {
        busy = false; // descarta este frame
      }
    }
  }
  scheduleNextFrame();
}

function drawOverlay(box) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!box) return;
  octx.strokeStyle = '#38bdf8';
  octx.lineWidth = 3;
  octx.strokeRect(box.x, box.y, box.w, box.h);
}

function showNoFace() {
  labelEl.textContent = 'nenhum rosto';
  labelEl.dataset.kind = 'none';
  stickyLabel = null;
  smoother.reset();
  barsEl.replaceChildren();
}

function updateEmotions(emotions) {
  const top = emotions[0];
  if (top && (top.confidence >= SWITCH_THRESHOLD || stickyLabel === null)) {
    stickyLabel = top.type;
  }
  labelEl.dataset.kind = 'face';
  labelEl.textContent = stickyLabel
    ? `${stickyLabel} · ${Math.round((top?.confidence ?? 0) * 100)}%`
    : '…';

  barsEl.replaceChildren(
    ...emotions.map((e) => {
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
    }),
  );
}

boot();
