// F5 — a inferencia de emocao (a parte cara: CNN) roda aqui, fora da main thread.
// Nunca chamamos session.run na main thread (relatorio 2.8).
//
// A deteccao facial (MediaPipe) fica na main thread — ver src/face-detect.js para o porque.
// Este worker recebe um ImageBitmap 224x224 por rosto (ja recortado pela main thread) e
// devolve o vetor de probabilidades bruto; a suavizacao temporal e o mapeamento p/ o
// contrato Rekognition acontecem na main thread (src/smoothing.js, src/emotions.js).
//
// Multiplos rostos: uma unica sessao ONNX processa os recortes em FILA (nao concorrente —
// session.run nao e seguro para chamadas sobrepostas na mesma sessao). Cada mensagem carrega
// o `id` do rosto (atribuido pelo FaceTracker na main thread) para a resposta poder ser
// roteada de volta ao rosto certo, mesmo com varios recortes pendentes.

import { initFer, classify } from './fer.js';

let ready = false;
let queue = Promise.resolve();

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
    const { id, bitmap } = msg;
    if (!ready) {
      bitmap.close?.();
      return;
    }
    // Encadeia no fim da fila em vez de `await` direto — assim mensagens seguintes sao
    // aceitas de imediato, mas a inferencia em si roda uma de cada vez.
    queue = queue.then(() => classifyOne(id, bitmap));
  }
};

async function classifyOne(id, bitmap) {
  try {
    const probs = await classify(bitmap);
    self.postMessage({ type: 'result', id, probs });
  } catch (err) {
    self.postMessage({ type: 'result', id, probs: null, error: String(err?.message || err) });
  } finally {
    bitmap.close?.();
  }
}
