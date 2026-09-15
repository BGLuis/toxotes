// F2 — wrapper do MediaPipe Face Detector (BlazeFace short-range).
//
// Roda na MAIN THREAD, nao no worker: o FilesetResolver do MediaPipe carrega o glue
// Emscripten (UMD, sem `export default`) por injecao de <script> / importScripts, o que
// nao existe num worker `type: module`. BlazeFace e barato (~1-3 ms), entao rodar a
// deteccao aqui nao trava a UI — o custo alto (CNN de emocao) fica no worker (F5).
//
// A runtime .wasm e servida de /mediapipe/wasm (ver vite.config.js e prepare-assets).
//
// Suporte a multiplos rostos: o FaceDetector do MediaPipe ja devolve TODAS as deteccoes
// acima de minDetectionConfidence (nao ha maxResults nessa API, ao contrario do
// ObjectDetector). MAX_FACES so limita quantos rostos this app tenta classificar por
// frame — cada rosto extra e mais uma inferencia sequencial no worker (ver worker.js).

import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

const MODEL_URL = '/models/blaze_face_short_range.tflite';
const WASM_ROOT = '/mediapipe/wasm';
const BOX_MARGIN = 0.2; // ~20% de folga p/ nao cortar testa/queixo (relatorio 2.2)
export const MAX_FACES = 4;

let detector = null;

/** @returns {Promise<'GPU'|'CPU'>} delegate efetivamente usado */
export async function initFaceDetector() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
  let lastErr;
  for (const delegate of ['GPU', 'CPU']) {
    try {
      detector = await FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO',
        minDetectionConfidence: 0.5,
      });
      return delegate;
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr;
}

function toBox(bb, srcW, srcH) {
  const mx = bb.width * BOX_MARGIN;
  const my = bb.height * BOX_MARGIN;
  const x = Math.max(0, Math.round(bb.originX - mx));
  const y = Math.max(0, Math.round(bb.originY - my));
  const w = Math.max(1, Math.min(srcW - x, Math.round(bb.width + 2 * mx)));
  const h = Math.max(1, Math.min(srcH - y, Math.round(bb.height + 2 * my)));
  return { x, y, w, h };
}

/**
 * Deteccao no frame atual do video. Modo VIDEO exige timestamp monotonico crescente (ms).
 * @param {HTMLVideoElement} source
 * @param {number} tsMs
 * @param {number} srcW - largura em pixels do frame (video.videoWidth)
 * @param {number} srcH
 * @returns {{ x: number, y: number, w: number, h: number }[]} caixas em pixels do frame,
 *   maiores primeiro, no maximo MAX_FACES (lista vazia se nenhum rosto)
 */
export function detectFace(source, tsMs, srcW, srcH) {
  const res = detector.detectForVideo(source, tsMs);
  const dets = res.detections;
  if (!dets || dets.length === 0) return [];

  return dets
    .slice()
    .sort((a, b) => {
      const areaA = a.boundingBox.width * a.boundingBox.height;
      const areaB = b.boundingBox.width * b.boundingBox.height;
      return areaB - areaA;
    })
    .slice(0, MAX_FACES)
    .map((d) => toBox(d.boundingBox, srcW, srcH));
}
