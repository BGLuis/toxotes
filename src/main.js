// Bootstrap na main thread: câmera (getUserMedia), detecção facial (MediaPipe BlazeFace),
// rastreamento de múltiplos rostos (IoU), estabilização temporal, histerese robusta de emoções,
// estimativa de idade (Rekognition AgeRange) e renderização na tela.

import './styles.css';
import { initFaceDetector, detectFace } from './face-detect.js';
import { toRekognition, toRekognitionAgeRange, EMOTION_METADATA, CANONICAL_EMOTION_ORDER } from './emotions.js';
import { EmaSmoother, ScalarSmoother } from './smoothing.js';
import { FaceTracker } from './face-tracker.js';

const video = document.getElementById('cam');
const stageEl = document.getElementById('stage');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const labelEl = document.getElementById('label');
const statusEl = document.getElementById('status');
const barsEl = document.getElementById('bars');

// Limiar e histerese para troca de rótulo sem flicker
const SWITCH_DELTA = 0.10;  // Vantagem mínima necessária sobre a classe fixada atual
const SWITCH_FRAMES = 6;    // Persistência mínima consecutiva (~200-300 ms)
const FACE_SIZE = 224;      // input do enet_b0_8_best_afew (EfficientNet-B0)
const FACE_COLORS = ['#38bdf8', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#fb923c'];

const tracker = new FaceTracker();

/**
 * @type {Map<number, {
 *   box: object,
 *   smoother: EmaSmoother,
 *   ageSmoother: ScalarSmoother,
 *   stickyLabel: string|null,
 *   candidateLabel: string|null,
 *   candidateFrames: number,
 *   ageRange: object|null,
 *   busy: boolean,
 *   emotions: object[]
 * }>}
 */
const faces = new Map();
const chipEls = new Map(); // id -> elemento .face-chip

let worker = null;
let faceDelegate = null;
let lastTs = 0;
let lastVisible = []; // último [{id, box}] visto

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

  setStatus('Carregando modelos (detecção + emoção + idade)…');
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
      setStatus(
        `Pronto — rosto em ${faceDelegate}, emoção em ${msg.ep.toUpperCase()}, idade em ${(msg.ageEp || 'WASM').toUpperCase()}.`,
        'ok'
      );
      scheduleNextFrame();
      break;
    case 'init-error':
      setStatus(`Erro ao iniciar a inferência: ${msg.error}`, 'error');
      break;
    case 'result': {
      const face = faces.get(msg.id);
      if (!face) break; // rosto já saiu de cena antes da resposta chegar — descarta
      face.busy = false;
      if (msg.error) console.warn('[fer] frame:', msg.id, msg.error);

      if (msg.age != null) {
        const smoothedAge = face.ageSmoother.push(msg.age);
        face.ageRange = toRekognitionAgeRange(smoothedAge);
      }

      if (msg.probs) {
        face.emotions = toRekognition(face.smoother.push(msg.probs));
        const top = face.emotions[0];
        if (top) {
          if (!face.stickyLabel) {
            face.stickyLabel = top.type;
            face.candidateLabel = null;
            face.candidateFrames = 0;
          } else if (top.type !== face.stickyLabel) {
            const stickyEntry = face.emotions.find((e) => e.type === face.stickyLabel);
            const stickyConf = stickyEntry ? stickyEntry.confidence : 0;
            if (top.confidence >= stickyConf + SWITCH_DELTA) {
              if (top.type === face.candidateLabel) {
                face.candidateFrames += 1;
                if (face.candidateFrames >= SWITCH_FRAMES) {
                  face.stickyLabel = top.type;
                  face.candidateLabel = null;
                  face.candidateFrames = 0;
                }
              } else {
                face.candidateLabel = top.type;
                face.candidateFrames = 1;
              }
            } else {
              face.candidateLabel = null;
              face.candidateFrames = 0;
            }
          } else {
            face.candidateLabel = null;
            face.candidateFrames = 0;
          }
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
    requestAnimationFrame(() => onFrame(performance.now()));
  }
}

function onFrame(now) {
  if (video.readyState >= 2) {
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
      if (!face || face.busy) continue;
      face.busy = true;
      sendFaceFrame(id, box);
    }
  }
  scheduleNextFrame();
}

function createFaceDom(id) {
  const group = document.createElement('div');
  group.className = 'face-group';

  const heading = document.createElement('div');
  heading.className = 'face-group-heading';
  heading.style.setProperty('--chip-color', colorForId(id));
  group.append(heading);

  const barRows = new Map();
  // As linhas são criadas rigorosamente na ordem canônica fixa
  for (const type of CANONICAL_EMOTION_ORDER) {
    const meta = EMOTION_METADATA[type] || { pt: type, icon: '' };
    const row = document.createElement('div');
    row.className = 'bar';

    const name = document.createElement('span');
    name.className = 'bar-name';
    name.textContent = `${meta.icon} ${meta.pt}`;

    const track = document.createElement('span');
    track.className = 'bar-track';
    const fill = document.createElement('span');
    fill.className = 'bar-fill';
    fill.style.width = '0%';
    track.append(fill);

    const val = document.createElement('span');
    val.className = 'bar-val';
    val.textContent = '0%';

    row.append(name, track, val);
    group.append(row);
    barRows.set(type, { row, fill, val });
  }

  return { group, heading, barRows };
}

function syncFaces(aliveIds, visible) {
  for (const [id, face] of faces.entries()) {
    if (!aliveIds.has(id)) {
      face.dom?.group.remove();
      faces.delete(id);
    }
  }
  for (const { id, box } of visible) {
    const existing = faces.get(id);
    if (existing) {
      existing.box = box;
    } else {
      faces.set(id, {
        box,
        smoother: new EmaSmoother(0.6),
        ageSmoother: new ScalarSmoother(0.85),
        stickyLabel: null,
        candidateLabel: null,
        candidateFrames: 0,
        ageRange: null,
        busy: false,
        emotions: [],
        dom: createFaceDom(id),
      });
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
    if (face) face.busy = false;
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

// Chips flutuantes por rosto, ancorados acima de cada caixa.
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
    if (face?.stickyLabel) {
      const activeEntry = face.emotions.find((e) => e.type === face.stickyLabel) || top;
      const activeConf = Math.round((activeEntry?.confidence ?? 0) * 100);
      const meta = EMOTION_METADATA[face.stickyLabel] || { pt: face.stickyLabel, icon: '🙂' };
      let text = `${meta.icon} ${meta.pt} · ${activeConf}%`;
      if (face.ageRange) {
        text += ` · 🎂 ${face.ageRange.formatted}`;
      }
      chip.textContent = text;
    } else {
      chip.textContent = '…';
    }
  }
  for (const [id, chip] of chipEls) {
    if (!seen.has(id)) {
      chip.remove();
      chipEls.delete(id);
    }
  }
}

function renderBars(visible) {
  if (visible.length === 0) {
    barsEl.replaceChildren();
    return;
  }

  const multi = visible.length > 1;
  const groupsToDisplay = [];

  for (const { id } of visible) {
    const face = faces.get(id);
    if (!face?.dom) continue;

    const { group, heading, barRows } = face.dom;
    if (multi) {
      heading.style.display = 'flex';
      const ageSuffix = face.ageRange ? ` · 🎂 ${face.ageRange.formatted}` : '';
      heading.textContent = `Rosto ${id}${ageSuffix}`;
    } else {
      heading.style.display = 'none';
    }

    const confMap = new Map((face.emotions || []).map((e) => [e.type, e.confidence]));
    const domType = face.stickyLabel;

    // Atualiza apenas os valores no lugar fixo de cada linha, sem jamais trocar a ordem
    for (const [type, { row, fill, val }] of barRows) {
      const conf = confMap.get(type) ?? 0;
      fill.style.width = `${(conf * 100).toFixed(1)}%`;
      val.textContent = `${Math.round(conf * 100)}%`;
      row.classList.toggle('is-top', type === domType);
    }

    groupsToDisplay.push(group);
  }

  const currentChildren = Array.from(barsEl.children);
  const changed =
    currentChildren.length !== groupsToDisplay.length ||
    groupsToDisplay.some((g, i) => g !== currentChildren[i]);

  if (changed) {
    barsEl.replaceChildren(...groupsToDisplay);
  }
}


boot();
