// F3 + F4 — pre-processamento e inferencia de emocao com ONNX Runtime Web.
// Modelo: enet_b0_8_best_afew (EfficientNet-B0), da biblioteca hsemotion-onnx
// (github.com/av-savchenko/hsemotion-onnx, codigo Apache-2.0; pesos publicados em
// github.com/HSE-asavchenko/face-emotion-recognition, treinados em AffectNet+AFEW+VGAF —
// dataset de treino de uso nao comercial; aceitavel aqui por este ser um projeto de
// demonstracao, nao um produto). Trocado a partir do emotion-ferplus (ver README, secao de
// licencas, e docs/reports/FER-NAVEGADOR-WASM.md).
//
// Contrato do modelo (confirmado lendo hsemotion_onnx/facial_emotions.py e o demo oficial):
//   input  : 1 tensor float32 "input", shape [1, 3, 224, 224] (NCHW), RGB
//            normalizado: pixel/255, depois (v - mean[c]) / std[c] por canal,
//            mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (estatistica ImageNet)
//   output : 1 tensor de scores nao normalizados (logits), 8 posicoes -> aplicar softmax
// Os nomes de input/output sao lidos da sessao em runtime (nao hardcoded).
//
// Recebe um ImageBitmap ja recortado e reescalado para 224x224 pela main thread
// (createImageBitmap com resizeWidth/resizeHeight). O demo oficial converte BGR->RGB antes
// do recorte; o getImageData do <canvas> ja devolve RGBA nessa mesma ordem, entao so
// descartamos o canal alfa — sem inverter canais.

import * as ort from 'onnxruntime-web';

// Servimos a runtime WebAssembly do proprio dominio, de /ort (copiada de node_modules por
// scripts/prepare-assets.mjs). Caminho explicito = sem depender da resolucao de asset do
// bundler dentro do Worker (que trava a instanciacao no build de producao). Nenhuma CDN.
ort.env.wasm.wasmPaths = '/ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 4)
  : 1;

const MODEL_URL = '/models/enet_b0_8_best_afew.onnx';
const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

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
  // Se o webgpu tropecar em algum operador (SiLU/squeeze-excite), cai para wasm.
  const warm = new ort.Tensor('float32', new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]);
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
 * @param {ImageBitmap} bitmap224 - rosto ja recortado e reescalado para 224x224
 * @returns {Promise<number[]>} 8 probabilidades (somam 1) na ordem de EMOTION_LABELS
 */
export async function classify(bitmap224) {
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.drawImage(bitmap224, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE); // RGBA, 0..255

  // NCHW: 3 planos separados (R, G, B — ordem ja compativel, ver contrato acima),
  // cada um normalizado com a estatistica ImageNet do modelo.
  const plane = SIZE * SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    input[p] = (data[i] / 255 - MEAN[0]) / STD[0];
    input[plane + p] = (data[i + 1] / 255 - MEAN[1]) / STD[1];
    input[2 * plane + p] = (data[i + 2] / 255 - MEAN[2]) / STD[2];
  }

  const tensor = new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]);
  const out = await session.run({ [inputName]: tensor });
  return softmaxIfNeeded(Array.from(out[outputName].data));
}

// O enet_b0_8_best_afew devolve logits. Mas defende contra um modelo que ja inclua softmax:
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
