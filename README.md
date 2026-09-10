# FER no navegador — WASM

Câmera aberta na tela + *label* dizendo a emoção do rosto. **100% client-side**: nenhum
*frame* sai da máquina, nenhum *backend*.

- **Detecção facial**: MediaPipe BlazeFace (*short-range*), *on-device*, na *main thread*.
- **Classificação de emoção**: [`emotion-ferplus`](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus)
  (ONNX Model Zoo, licença **MIT**), quantizado INT8, via **ONNX Runtime Web** num **Web Worker**,
  em WebGPU quando disponível e WebAssembly (SIMD + *threads*) como *fallback*.
- **Saída** no formato do Amazon Rekognition `DetectFaces`: `[{ type, confidence }]` ordenado
  por confiança. O Rekognition entra **só como referência de contrato** — nenhuma chamada à AWS.

Implementa [`docs/reports/FER-NAVEGADOR-WASM.md`](docs/reports/FER-NAVEGADOR-WASM.md).

## Rodar

Requer Node 18+ e um navegador com suporte a `getUserMedia` (Chrome/Edge recomendados; é onde
o *execution provider* `webgpu` funciona *out-of-box*).

```bash
npm install
npm run dev      # http://localhost:5173
```

`npm install` baixa as dependências; `predev`/`prebuild` copiam a runtime `.wasm` do
`onnxruntime-web` e do MediaPipe de `node_modules/` para `public/ort/` e `public/mediapipe/`
(elas **não** podem vir de CDN sob `COEP: require-corp`).

Produção:

```bash
npm run build    # -> dist/  (postbuild remove a cópia órfã da runtime .wasm em dist/assets/)
npm run preview  # serve dist/ com os headers COOP/COEP
```

### `getUserMedia` exige *secure context*

`http://localhost` (dev) ou `https://` (produção). Numa origem `http://` que não seja
`localhost`, `navigator.mediaDevices` é `undefined` e a página mostra o aviso correspondente.

### COOP/COEP / *cross-origin isolation*

As *threads* do WASM exigem `SharedArrayBuffer`, que exige `crossOriginIsolated === true`, que
exige os headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

- **Dev/preview**: emitidos por `vite.config.js`.
- **Deploy estático**: `public/_headers` (Cloudflare Pages / Netlify). **GitHub Pages não
  suporta headers customizados** — não serve para este projeto. Sem esses headers a página
  ainda roda, mas o `onnxruntime-web` cai para *single-thread*.

Confirme no *console*: `crossOriginIsolated` deve ser `true`.

## Taxonomia — 7 classes

O `emotion-ferplus` tem 8 saídas (FER+ = FER-2013 + `contempt`). Expomos **7**, mapeadas para
o vocabulário do Rekognition; `contempt` é descartado e a distribuição é renormalizada
(`src/emotions.js`). Nunca emitimos `CONFUSED` nem `UNKNOWN`.

| Rekognition | origem FER+ |
|---|---|
| `HAPPY` | `happiness` |
| `SAD` | `sadness` |
| `ANGRY` | `anger` |
| `SURPRISED` | `surprise` |
| `DISGUSTED` | `disgust` |
| `FEAR` | `fear` |
| `CALM` | `neutral` (aproximação) |

Sem rosto no *frame*: *label* = "nenhum rosto", lista vazia, inferência pulada.

## Arquitetura

```
main thread                              Web Worker (módulo ES)
───────────                              ─────────────────────
getUserMedia → <video>
requestVideoFrameCallback loop
  ├─ MediaPipe FaceDetector (barato)
  ├─ desenha a bounding box no <canvas>
  ├─ createImageBitmap(video, recorte, →64×64)
  │        └── postMessage(bitmap, [transfer]) ──────►  fer.js
  │                                                      ├─ OffscreenCanvas → cinza (luma)
  │                                                      ├─ ort.Tensor [1,1,64,64]
  │                                                      ├─ session.run  (webgpu → wasm)
  │                                                      └─ softmax
  │        ◄──────────────── postMessage({ probs }) ─────┘
  ├─ EMA (suavização temporal, src/smoothing.js)
  ├─ mapeia p/ contrato Rekognition (src/emotions.js)
  └─ histerese de label (só troca com confiança ≥ 0,4) + barras
```

**Por que a detecção facial na *main thread* e não no worker** — o *report* pedia o
*pipeline* inteiro no worker. O `FilesetResolver` do MediaPipe carrega o *glue* Emscripten
(UMD, sem `export default`) por injeção de `<script>` / `importScripts`, o que não existe num
worker `type: module`; num worker de módulo ele falha com `ModuleFactory not set`. BlazeFace
custa ~1–3 ms, então rodá-lo na *main thread* não trava a UI. **A regra que importa —
`session.run` (a CNN de emoção) nunca na *main thread* — continua valendo**: essa parte está
no worker.

## Custo de *download* (medido neste projeto)

Primeira visita, depois **cache-first** via *service worker* (`public/sw.js`) para
`/models`, `/ort`, `/mediapipe`:

| Artefato | Tamanho | Comprimido (gzip) |
|---|---|---|
| `emotion-ferplus-int8.onnx` | 19 MB | ~18 MB (já quantizado) |
| runtime ONNX — `ort-wasm-simd-threaded.jsep.wasm` (caminho webgpu) | 27 MB | ~6,5 MB |
| runtime ONNX — `ort-wasm-simd-threaded.wasm` (caminho wasm) | 14 MB | ~5 MB |
| MediaPipe `vision_wasm_internal.wasm` | 11 MB | ~4 MB |
| `blaze_face_short_range.tflite` | 228 KB | — |
| app (JS+CSS+worker) | ~550 KB | ~180 KB |

Só **uma** das runtimes ONNX é buscada em tempo de execução (jsep se há adapter WebGPU,
senão a `wasm`). O modelo de emoção é o gargalo de *download* — trocar por um FER menor com
licença compatível é a otimização óbvia (ver *report*, risco 1).

## Verificação — pendente (é manual e gráfica)

Ver `docs/reports/FER-NAVEGADOR-WASM.md` seção 5. Estado atual:

- [x] *Bundle* não referencia CDN externa; `.wasm`/`.onnx` servidos da própria origem.
- [x] Servidor declara COOP/COEP; `crossOriginIsolated === true` no Chrome.
- [x] Taxonomia expõe 7 tipos, nunca `CONFUSED`/`UNKNOWN`.
- [x] Câmera abre, permissão negada mostra mensagem (não tela quebrada).
- [x] *Pipeline* de emoção no Worker; *label* atualiza ao mudar a expressão; *zero*
      requisições de vídeo/imagem saindo da página (validado com câmera sintética + Chrome
      DevTools; ver histórico de implementação).
- [ ] **Rosto real**, luz de escritório, óculos, várias etnias — não validado.
- [ ] **WebGPU de verdade** (máquina com GPU/adapter) — não validado aqui (ambiente de teste
      sem adapter caiu para `wasm`, como esperado).
- [ ] Firefox / Safari (Technology Preview) — não validado.
- [ ] ≥ ~10 FPS numa máquina de dev típica — não medido.

## Uso responsável

Reconhecimento de emoções tem **restrição regulatória e ética** (o EU AI Act classifica
*emotion recognition* como de risco e restringe uso em trabalho e educação). A saída reflete
**expressão facial**, não estado emocional interno. Antes de adotar em produto, valide o
contexto legal — ver *report*, risco 4.

## Licenças dos modelos

- `emotion-ferplus-int8.onnx` — ONNX Model Zoo, **MIT**.
- `blaze_face_short_range.tflite` — Google MediaPipe, **Apache-2.0**.
