// F5 — a inferencia de emocao (a parte cara: CNN) roda aqui, fora da main thread.
// Nunca chamamos session.run na main thread (relatorio 2.8).
//
// A deteccao facial (MediaPipe) fica na main thread — ver src/face-detect.js para o porque.
// Este worker recebe um ImageBitmap 64x64 (rosto ja recortado) e devolve o vetor de
// probabilidades bruto; a suavizacao temporal e o mapeamento p/ o contrato Rekognition
// acontecem na main thread (src/smoothing.js, src/emotions.js).

import { initFer, classify } from './fer.js';

let ready = false;

self.onmessage = async (ev) => {
  const msg = ev.data;

  if (msg.type === 'init') {
    try {
      const ep = await initFer();
      ready = true;
      self.postMessage({ type: 'ready', ep });
    } catch (err) {
      self.postMessage({ type: 'init-error', error: String(err?.message || err) });
    }
    return;
  }

  if (msg.type === 'frame') {
    const { bitmap } = msg;
    if (!ready) {
      bitmap.close?.();
      return;
    }
    try {
      const probs = await classify(bitmap);
      self.postMessage({ type: 'result', probs });
    } catch (err) {
      self.postMessage({ type: 'result', probs: null, error: String(err?.message || err) });
    } finally {
      bitmap.close?.();
    }
  }
};
