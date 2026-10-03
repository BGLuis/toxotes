// Inferência neural (FER + Idade) executada fora da main thread.
//
// A detecção facial (MediaPipe) fica na main thread (src/face-detect.js).
// Este worker recebe um ImageBitmap por rosto (já recortado pela main thread) e executa:
//   1. Emoção (enet_b0_8_best_afew): a cada frame recebido.
//   2. Idade (age-v1.onnx): amortizada a cada ~15 frames (~1 Hz) por rosto.
//
// Múltiplos rostos: uma única sessão ONNX processa os recortes em FILA (não concorrente).
// Cada mensagem carrega o `id` do rosto (atribuído pelo FaceTracker na main thread) para
// a resposta poder ser roteada de volta ao rosto certo.

import { initFer, classify } from './fer.js';
import { initAge, classifyAge } from './age.js';

let ready = false;
let queue = Promise.resolve();
const ageByFaceId = new Map();
const frameCounts = new Map();
const AGE_INTERVAL_FRAMES = 15; // ~1 Hz

self.onmessage = async (ev) => {
  const msg = ev.data;

  if (msg.type === 'init') {
    try {
      const [ferEp, ageEp] = await Promise.all([initFer(), initAge()]);
      ready = true;
      self.postMessage({ type: 'ready', ep: ferEp, ageEp });
    } catch (err) {
      self.postMessage({ type: 'init-error', error: String(err?.message || err) });
    }
    return;
  }

  if (msg.type === 'frame') {
    const { id, bitmap } = msg;
    if (!ready) {
      bitmap.close?.();
      return;
    }
    queue = queue.then(() => classifyOne(id, bitmap));
  }
};

async function classifyOne(id, bitmap) {
  try {
    const probs = await classify(bitmap);

    const count = (frameCounts.get(id) || 0) + 1;
    frameCounts.set(id, count);

    if (!ageByFaceId.has(id) || count % AGE_INTERVAL_FRAMES === 0) {
      try {
        const age = await classifyAge(bitmap);
        ageByFaceId.set(id, age);
      } catch (ageErr) {
        console.warn('[worker] erro na estimativa de idade:', ageErr);
      }
    }

    self.postMessage({ type: 'result', id, probs, age: ageByFaceId.get(id) ?? null });
  } catch (err) {
    self.postMessage({ type: 'result', id, probs: null, age: null, error: String(err?.message || err) });
  } finally {
    bitmap.close?.();
  }
}
