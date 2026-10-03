// Inferência de estimativa de idade com ONNX Runtime Web.
// Modelo: age-v1.onnx (Cydral / Dlib / mowshon, licença CC0-1.0).
//
// Contrato do modelo:
//   input  : 1 tensor float32, shape [1, 3, 64, 64], RGB em NCHW.
//            Normalização: (RGB_byte - channel_mean) / 256.0
//            médias: R=122.781998, G=117.000999, B=104.297997
//   output : 1 tensor float32 com 81 probabilidades (softmax embutido no grafo).
//            Pesos das classes: bin 0 = 0.25; bins 1..80 = 1..80.
//            Idade estimada = soma ponderada sum(prob[i] * weight[i]).

import * as ort from 'onnxruntime-web';

const MODEL_URL = '/models/age-v1.onnx';
const SIZE = 64;

// Médias por canal especificadas no manifesto de treinamento
const MEAN_R = 122.781998;
const MEAN_G = 117.000999;
const MEAN_B = 104.297997;

// Pesos dos 81 bins de idade
const AGE_WEIGHTS = new Float32Array(81);
AGE_WEIGHTS[0] = 0.25;
for (let i = 1; i <= 80; i += 1) {
  AGE_WEIGHTS[i] = i;
}

let session = null;
let inputName = null;
let outputName = null;
let activeEp = 'wasm';

/** @type {OffscreenCanvas} */
let canvas = null;
/** @type {OffscreenCanvasRenderingContext2D} */
let ctx = null;

function createSession(executionProviders) {
  return ort.InferenceSession.create(MODEL_URL, {
    executionProviders,
    graphOptimizationLevel: 'all',
  });
}

/** @returns {Promise<'webgpu'|'wasm'>} execution provider ativo */
export async function initAge() {
  let hasWebGpuAdapter = false;
  try {
    const adapter = await Promise.race([
      navigator.gpu?.requestAdapter?.() ?? Promise.resolve(null),
      new Promise((resolve) => setTimeout(() => resolve(null), 1500)),
    ]);
    hasWebGpuAdapter = Boolean(adapter);
  } catch {
    hasWebGpuAdapter = false;
  }

  const plan = hasWebGpuAdapter
    ? [
        { ep: 'webgpu', list: ['webgpu', 'wasm'] },
        { ep: 'wasm', list: ['wasm'] },
      ]
    : [{ ep: 'wasm', list: ['wasm'] }];

  let lastErr;
  for (const step of plan) {
    try {
      session = await createSession(step.list);
      activeEp = step.ep;
      break;
    } catch (err) {
      lastErr = err;
      session = null;
    }
  }
  if (!session) throw lastErr;

  inputName = session.inputNames[0] || 'images';
  outputName = session.outputNames[0] || 'probabilities';

  canvas = new OffscreenCanvas(SIZE, SIZE);
  ctx = canvas.getContext('2d', { willReadFrequently: true });

  // Warm-up
  const warm = new ort.Tensor('float32', new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]);
  try {
    await session.run({ [inputName]: warm });
  } catch (err) {
    if (activeEp === 'wasm') throw err;
    session = await createSession(['wasm']);
    activeEp = 'wasm';
    inputName = session.inputNames[0] || 'images';
    outputName = session.outputNames[0] || 'probabilities';
    await session.run({ [inputName]: warm });
  }

  return activeEp;
}

/**
 * @param {ImageBitmap} bitmap64 - rosto recortado 64x64
 * @returns {Promise<number>} idade escalar estimada em anos
 */
export async function classifyAge(bitmap64) {
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.drawImage(bitmap64, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE); // RGBA

  const numPixels = SIZE * SIZE;
  const input = new Float32Array(3 * numPixels);
  const offsetG = numPixels;
  const offsetB = 2 * numPixels;

  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    input[p] = (data[i] - MEAN_R) / 256.0;
    input[offsetG + p] = (data[i + 1] - MEAN_G) / 256.0;
    input[offsetB + p] = (data[i + 2] - MEAN_B) / 256.0;
  }

  const tensor = new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]);
  const out = await session.run({ [inputName]: tensor });
  const probs = out[outputName].data;

  let expectedAge = 0;
  for (let i = 0; i < probs.length && i < AGE_WEIGHTS.length; i += 1) {
    expectedAge += probs[i] * AGE_WEIGHTS[i];
  }

  return expectedAge;
}
