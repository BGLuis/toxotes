// F3 + F4 — pre-processamento e inferencia de emocao com ONNX Runtime Web.
// Modelo: emotion-ferplus (ONNX Model Zoo, MIT), quantizado INT8.
//
// Contrato do modelo (emotion-ferplus):
//   input  : 1 tensor float32, shape [1, 1, 64, 64], escala de cinza em [0, 255] BRUTA
//            (a subtracao de media esta embutida no grafo — nao normalizar aqui)
//   output : 1 tensor de scores nao normalizados (logits), 8 posicoes -> aplicar softmax
// Os nomes de input/output sao lidos da sessao em runtime (variam no modelo quantizado).
//
// Recebe um ImageBitmap ja recortado e reescalado para 64x64 pela main thread
// (createImageBitmap com resizeWidth/resizeHeight). Aqui so falta a conversao p/ cinza.

import * as ort from 'onnxruntime-web';

// Servimos a runtime WebAssembly do proprio dominio, de /ort (copiada de node_modules por
// scripts/prepare-assets.mjs). Caminho explicito = sem depender da resolucao de asset do
// bundler dentro do Worker (que trava a instanciacao no build de producao). Nenhuma CDN.
ort.env.wasm.wasmPaths = '/ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 4)
  : 1;

const MODEL_URL = '/models/emotion-ferplus-int8.onnx';
const SIZE = 64;

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

/** @returns {Promise<'webgpu'|'wasm'>} execution provider efetivamente ativo */
export async function initFer() {
  // So oferece webgpu ao ORT se um adapter WebGPU real existir — senao o ORT aceita a
  // sessao e cai para wasm em silencio, e o EP reportado fica mentiroso.
  let hasWebGpuAdapter = false;
  try {
    const adapter = await Promise.race([
      navigator.gpu?.requestAdapter?.() ?? Promise.resolve(null),
      new Promise((resolve) => setTimeout(() => resolve(null), 1500)), // nao deixa travar o boot
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

  inputName = session.inputNames[0];
  outputName = session.outputNames[0];

  canvas = new OffscreenCanvas(SIZE, SIZE);
  ctx = canvas.getContext('2d', { willReadFrequently: true });

  // Warm-up: compila os kernels e confirma que o EP escolhido roda o grafo de fato.
  // Se o webgpu tropecar em algum operador quantizado, cai para wasm.
  const warm = new ort.Tensor('float32', new Float32Array(SIZE * SIZE), [1, 1, SIZE, SIZE]);
  try {
    await session.run({ [inputName]: warm });
  } catch (err) {
    if (activeEp === 'wasm') throw err;
    session = await createSession(['wasm']);
    activeEp = 'wasm';
    inputName = session.inputNames[0];
    outputName = session.outputNames[0];
    await session.run({ [inputName]: warm });
  }

  return activeEp;
}

/**
 * @param {ImageBitmap} bitmap64 - rosto ja recortado e reescalado para 64x64
 * @returns {Promise<number[]>} 8 probabilidades (somam 1) na ordem de FERPLUS_LABELS
 */
export async function classify(bitmap64) {
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.drawImage(bitmap64, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE); // RGBA, 0..255

  const input = new Float32Array(SIZE * SIZE);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    // luma BT.601 — cinza bruto, sem dividir por 255 (ver contrato acima)
    input[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  const tensor = new ort.Tensor('float32', input, [1, 1, SIZE, SIZE]);
  const out = await session.run({ [inputName]: tensor });
  return softmaxIfNeeded(Array.from(out[outputName].data));
}

// O emotion-ferplus devolve logits. Mas defende contra um modelo que ja inclua softmax:
// se a saida ja parece distribuicao (tudo em [0,1] e soma ~1), nao aplica de novo.
function softmaxIfNeeded(scores) {
  const sum = scores.reduce((a, b) => a + b, 0);
  const looksNormalized = scores.every((v) => v >= 0 && v <= 1) && Math.abs(sum - 1) < 0.05;
  if (looksNormalized) return scores;

  const max = Math.max(...scores);
  const exps = scores.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}
