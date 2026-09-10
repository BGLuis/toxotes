# FER no navegador — proposta de implementação 100% client-side com WASM

| Campo | Valor |
|-------|-------|
| **Status** | ✅ Implementado (F0–F6). Verificação com rosto real / cross-browser / FPS: pendente |
| **Cobertura** | 7 de 7 fases andaimadas e funcionando *end-to-end* (câmera sintética + Chrome DevTools) |
| **Esforço [modelado]** | 4,5–7 dias-dev no escopo completo; 1–1,5 d no mínimo viável |
| **Depende de** | ~~Um modelo FER pré-treinado com licença compatível~~ → **resolvido**: `emotion-ferplus` (ONNX Model Zoo, MIT) |
| **Fecha requisito** | Câmera aberta na tela + *label* dizendo a emoção, sem *backend* |

> **Atualização pós-implementação.** O código vive na raiz do projeto; como rodar e o que foi
> (e não foi) verificado estão em [`../../README.md`](../../README.md). Decisões que
> divergiram deste relatório durante a implementação:
>
> 1. **Modelo (risco 1, seção 6).** Adotado o `emotion-ferplus` do ONNX Model Zoo, licença
>    **MIT** — o risco crítico de licença não se materializou. Ele tem 8 classes (FER+ =
>    FER-2013 + `contempt`); expomos as 7 mapeáveis ao Rekognition e descartamos `contempt`
>    renormalizando (decisão da seção 2.7 mantida: sem `CONFUSED`/`UNKNOWN` inventados).
> 2. **Detecção facial na *main thread*, não no worker (seção 2.6 / F5).** O `FilesetResolver`
>    do MediaPipe carrega *glue* Emscripten UMD por `<script>`/`importScripts`, incompatível
>    com um worker `type: module` (`ModuleFactory not set`). BlazeFace custa ~1–3 ms — não
>    trava a UI. **A regra dura da seção 2.8 continua respeitada: `session.run` (emoção) só
>    no worker.**
> 3. **Runtime `.wasm` do ONNX servida de `/ort` (`ort.env.wasm.wasmPaths`), não via
>    resolução de *asset* do *bundler*** — esta última travava a instanciação no *build* de
>    produção dentro do Worker.

---

## 1. Estado atual — evidências

### O que existe

Nada. O diretório de trabalho está vazio:

```bash
ls -la /home/luis/Documents/hand-on/toxotes
# → total 0
#   drwxr-xr-x 1 luis luis 0 .
#   drwxr-xr-x 1 luis luis … ..
```

Não é um repositório git — não há *branch* nem *commit* de referência para ancorar a análise:

```bash
git -C /home/luis/Documents/hand-on/toxotes status
# → fatal: not a git repository (or any of the parent directories): .git
```

Não há `src/`, `package.json`, `index.html` nem `docs/` (a pasta `docs/reports/` foi criada
agora, apenas para receber este arquivo):

```bash
find /home/luis/Documents/hand-on/toxotes -mindepth 1 -not -path '*/docs/reports*'
# → 0 resultados
```

O relatório é, portanto, **integralmente prospectivo**. Não há código a auditar, não há
métrica de *bundle* a extrair, não há trace a ler. Todo número de esforço e de tamanho de
artefato adiante carrega o marcador `[modelado]`.

### O contrato de referência — Amazon Rekognition

O pedido cita "baseado no Amazon Rekognition". Fica decidido: o Rekognition entra **apenas
como espelho do contrato de saída** — a taxonomia de emoções e o formato da resposta que a
implementação local deve reproduzir. **Nenhuma chamada à AWS** é feita; enviar *frames* de
vídeo a um serviço de nuvem contradiz o requisito central (sem *backend*, privacidade total).

O `DetectFaces` da API do Rekognition, quando `Attributes` inclui `EMOTIONS`, devolve por
rosto um array `Emotions[]` de objetos `{ Type, Confidence }`. O enum de `Type` documentado é:

```
HAPPY · SAD · ANGRY · CONFUSED · DISGUSTED · SURPRISED · CALM · FEAR · UNKNOWN
```

A própria documentação da AWS adverte que o resultado é uma predição a partir da **expressão
facial** e **não** uma determinação do estado emocional interno da pessoa — ver seção 6,
risco 4. O alvo de engenharia é reproduzir a **forma** dessa saída (`[{ type, confidence }]`
ordenado por confiança), não o serviço.

---

## 2. As 8 decisões de design

### 2.1 Onde roda a inferência

| Opção | O que é | Custo | Veredito |
|---|---|---|---|
| Navegador (WASM/WebGPU) | `getUserMedia` → modelo local; nada sai da máquina | Modelo baixado uma vez (~2–13 MB `[modelado]`); usa CPU/GPU do cliente | **Recomendada** — privacidade total, custo zero de GPU na nuvem, latência de rede nula |
| Nuvem (Rekognition ou modelo próprio em servidor) | *Frames* enviados a um endpoint; inferência remota | Custo por chamada, infra de GPU, latência de *round-trip*, exposição de vídeo | Rejeitada: contradiz o requisito "sem *backend*" e o motivo do pedido |

O Rekognition permanece **só como referência de contrato** (2.7). A inferência é 100%
client-side.

### 2.2 Captura e pré-processamento

O fluxo é fixo, independentemente da trilha escolhida em 2.4:

1. `navigator.mediaDevices.getUserMedia({ video: true, audio: false })` → `stream` num
   elemento `<video autoplay playsinline muted>`.
2. A cada *frame* processado, desenhar o `<video>` num `<canvas>` obtido com
   `getContext('2d', { willReadFrequently: true })` — sem essa *flag*, `getImageData` força
   leitura GPU→CPU repetida e derruba o FPS.
3. Recortar a *bounding box* do rosto (origem em 2.3), com uma margem de ~20% para não cortar
   testa/queixo.
4. Redimensionar o recorte para o *input shape* do modelo de emoção — `48×48` ou `64×64` em
   tons de cinza para arquiteturas tipo Mini-Xception treinadas em FER-2013; `224×224` RGB
   para variantes MobileNet-FER.
5. Normalizar **exatamente como no treino** do modelo: alguns esperam `[0,1]` (`pixel/255`),
   outros `[-1,1]`, outros subtração de média/desvio por canal. Errar isto degrada a
   acurácia em silêncio — ver seção 4.

### 2.3 Detecção facial

| Opção | O que é | Custo | Veredito |
|---|---|---|---|
| MediaPipe Face Detector (`@mediapipe/tasks-vision`) | Detector BlazeFace empacotado nas *Tasks Vision* da Google; roda *on-device*; devolve *bounding box* + 6 *keypoints* (olhos, nariz, boca, trágions) | +1 runtime WASM (~1–3 MB `[modelado]`); API própria de `FilesetResolver` | **Recomendada** — recorte preciso e *keypoints* permitem alinhamento da face antes da classificação |
| BlazeFace via TensorFlow.js | Mesmo modelo-base, empacotado no ecossistema TFJS | Arrasta TFJS inteiro se ele já não estiver no *bundle* | Rejeitada: sem vantagem sobre MediaPipe se a trilha B (2.4) não usa TFJS |
| SSD-Mobilenet do `face-api.js` | Detector embutido na Trilha A | Já incluso na Trilha A; nenhuma dependência extra | Aceitável **apenas** dentro da Trilha A |

Alinhar a face pelos *keypoints* (rotacionar para deixar os olhos na horizontal) melhora a
acurácia de qualquer classificador FER treinado em faces frontais. É opcional no mínimo
viável, recomendado no escopo completo.

### 2.4 Pilha de classificação de emoções — as duas trilhas

Este é o núcleo do pedido. As duas trilhas são viáveis; diferem em esforço inicial, controle
sobre o modelo e teto de performance.

| Aspecto | Trilha A — `@vladmandic/face-api` | Trilha B — ONNX Runtime Web + MediaPipe |
|---|---|---|
| O que é | Detecção facial + classificador de expressões num pacote único, sobre backend TensorFlow.js (WASM ou WebGL) | MediaPipe recorta a face (2.3); um modelo `.onnx` de FER roda no `onnxruntime-web` |
| Esforço de *setup* [modelado] | Mínimo — 1 dependência, modelos pré-hospedados, API única (`detectAllFaces().withFaceExpressions()`) | Maior — 2 runtimes para orquestrar, obter/converter o modelo FER, casar o pré-processamento com o treino |
| Controle sobre o modelo | Baixo — preso ao *expression net* embutido (7 classes fixas), sem *swap* trivial | Alto — troca o `.onnx` livremente, quantiza INT8/FP16, escolhe arquitetura (Mini-Xception, MobileNetV2-FER, EmoNet…) |
| Aceleração | WASM (SIMD/threads) ou WebGL; **sem WebGPU de primeira classe** | *Execution provider* `webgpu` dedicado, com *fallback* `wasm` SIMD+threads (2.5) |
| Manutenção | `@vladmandic/face-api` **1.7.15**, sem *release* há ~2 anos; TFJS como *peer dependency* que precisa ser fixada | `onnxruntime-web` ~**1.26.0** (mai/2026), ativo; `@mediapipe/tasks-vision` ~**0.10.x**, ativo |
| Download total [modelado] | ~6–13 MB — SSD-Mobilenet + *expression net* + *core* do TFJS | ~2–8 MB — detector MediaPipe + FER quantizado — mais ~1–3 MB do runtime `.wasm`/`.jsep.wasm` |
| Prós | Caminho mais curto ao primeiro resultado na tela; superfície de API pequena; muitos exemplos prontos; detecção e emoção num só passo | Melhor teto de performance (WebGPU joga a convolução nos *shaders*); modelo substituível e quantizável; runtime sob manutenção ativa; *pipeline* explícito e depurável peça a peça; alinhado ao que o pedido descreve (WASM + WebGPU) |
| Contras | Dependência estagnada — sem correção de bugs nem novos *targets*; acoplada ao TFJS; sem WebGPU; trocar o modelo de emoção é reescrever a biblioteca; *bundle* maior | Mais peças para integrar e versionar; risco real de *mismatch* de pré-processamento entre o `.onnx` e o código (seção 4); é preciso encontrar/converter um `.onnx` de FER com licença adequada (seção 6, risco 1); curva de aprendizado do `onnxruntime-web` (nomes de *input*/*output*, *execution providers*, servir os `.wasm`) |
| Veredito | Aceitável para **protótipo / prova de conceito** rápida | **Recomendada** — alinhada ao requisito (WASM + WebGPU), com runtime mantido e modelo sob controle |

**Decisão:** seguir a **Trilha B** como caminho principal. Manter a Trilha A documentada como
rota de *fallback* caso a fase F4 (obter um `.onnx` de FER utilizável) atrase — ela entrega
"algo na tela" em ~1 dia `[modelado]`, ao custo de teto de performance e manutenção.

### 2.5 Execution provider do `onnxruntime-web`

| Opção | O que é | Custo | Veredito |
|---|---|---|---|
| `webgpu` | EP sobre a WebGPU API; convolução nos *shaders* da GPU | Exige `ort-wasm-simd-threaded.jsep.wasm` servido junto; suporte *out-of-box* só em Chrome/Edge (Firefox atrás de *flag*; Safari em *Technology Preview* na data deste relatório) | **Recomendada como primária**, dentro de uma lista de *fallback* |
| `wasm` (SIMD + threads) | Inferência na CPU via WebAssembly com *multi-threading* | *Threads* exigem `SharedArrayBuffer` → exigem *headers* COOP/COEP (2.6, seção 4); sem eles, cai para *single-thread* | **Recomendada como *fallback*** — universal, sempre presente |
| `webgl` | EP legado sobre WebGL | Cobertura de operadores menor que `webgpu`; em depreciação gradual | Rejeitada: `webgpu` o substitui onde há GPU; `wasm` cobre o resto |
| `webnn` | EP sobre a Web Neural Network API | Disponibilidade e cobertura de *hardware* ainda irregulares | Rejeitada por ora — reavaliar quando o suporte estabilizar |

Configuração recomendada:

```js
// lista de fallback: tenta webgpu, cai para wasm
const session = await ort.InferenceSession.create('./models/fer.onnx', {
  executionProviders: ['webgpu', 'wasm'],
});
```

### 2.6 Thread e concorrência

Rodar o *pipeline* de visão na *main thread* trava a UI a cada `session.run` — o *label*
congela, a página perde resposta. O *pipeline* inteiro (detecção MediaPipe + pré-processamento
+ inferência ONNX) deve viver num **Web Worker**:

- `<video>` e `<canvas>` de exibição ficam na *main thread*; o *frame* vai ao Worker como
  `ImageBitmap` (transferível, cópia zero) via `postMessage`.
- O Worker devolve `[{ type, confidence }]`; a *main thread* só atualiza o texto do *label*.
- *Threads* do WASM (tanto no EP `wasm` quanto no `.jsep.wasm` do `webgpu`) exigem
  `SharedArrayBuffer`, que exige `crossOriginIsolated === true`, que exige os *headers*:

  ```
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  ```

- O disparo por *frame* deve usar `video.requestVideoFrameCallback()` em vez de
  `setInterval(…, 200)` — abas em segundo plano sofrem *throttling* de *timers* e o
  `setInterval` para de disparar de forma previsível.

### 2.7 Taxonomia de emoções

O modelo FER público típico (treinado em **FER-2013**) tem **7 classes**:

```
angry · disgust · fear · happy · sad · surprise · neutral
```

O contrato do Rekognition tem **8** (`+ UNKNOWN`). A divergência é estrutural:

| Rekognition | FER-2013 | Mapeamento |
|---|---|---|
| `HAPPY` | `happy` | direto |
| `SAD` | `sad` | direto |
| `ANGRY` | `angry` | direto |
| `SURPRISED` | `surprise` | direto |
| `DISGUSTED` | `disgust` | direto |
| `FEAR` | `fear` | direto |
| `CALM` | `neutral` | aproximação — `neutral` ≈ `CALM` |
| `CONFUSED` | — | **não existe no FER-2013**; não inventar |

**Decisão:** expor as **7 classes do FER-2013** na saída, no formato do Rekognition
(`[{ type, confidence }]` ordenado por confiança desc.), rotulando `neutral` como `CALM` para
aproximar o vocabulário. Não fabricar `CONFUSED` nem `UNKNOWN` a partir de nada — se nenhuma
face for detectada, a saída é uma lista vazia e o *label* diz "nenhum rosto".

### 2.8 O que não fazer

- **Não** enviar *frames* a nenhum serviço remoto — anula o motivo do projeto.
- **Não** rodar `session.run` (nem a detecção) na *main thread*.
- **Não** assumir `SharedArrayBuffer` sem checar `crossOriginIsolated` e ter *fallback*
  *single-thread*.
- **Não** normalizar o *tensor* de forma diferente do treino do modelo.
- **Não** aplicar `softmax` sobre uma saída que já é probabilidade — nem tratar *logits*
  crus como confiança.
- **Não** hospedar numa origem que não emite COOP/COEP se as *threads* WASM forem
  necessárias (seção 4).
- **Não** trocar o *label* a cada *frame* sem suavização temporal — vira ruído visual (F6).

---

## 3. Plano de implementação

| Fase | Conteúdo | Esforço [modelado] |
|---|---|---|
| **F0** | Andaime do projeto (`index.html`, `package.json`, *bundler*); servidor estático emitindo COOP/COEP; acesso via `https://` ou `localhost` (pré-requisito de `getUserMedia`) | 0,5 d |
| **F1** | `getUserMedia` → `<video>` visível + `<canvas>`; *label* estático "carregando…"; tratamento de permissão negada | 0,5 d |
| **F2** | Integrar MediaPipe Face Detector na *main thread*; desenhar a *bounding box* sobre o vídeo; caso "nenhuma face" | 0,5–1 d |
| **F3** | Recorte da face + *resize* + conversão para *tensor* `Float32Array` no *shape* e na normalização do modelo escolhido | 0,5 d |
| **F4** | Obter/converter um `.onnx` de FER com licença compatível (seção 6); `InferenceSession.create` com EP `wasm`; `session.run`; *argmax* + limiar de confiança; *label* dinâmico | 1–1,5 d |
| **F5** | Mover detecção + pré-processamento + inferência para um Web Worker (`ImageBitmap` transferível); ativar EP `['webgpu','wasm']`; servir os `.wasm`/`.jsep.wasm` | 1–1,5 d |
| **F6** | Suavização temporal (EMA ou média móvel das probabilidades); *cache* do modelo em `Cache Storage`; polir UI (espelhar vídeo, estados de erro) | 0,5–1 d |

**Mínimo viável** (só o que foi pedido literalmente — câmera + *label*): F0–F4, *pipeline* na
*main thread*, apenas EP `wasm` ≈ **1–1,5 d** `[modelado]`.

**Escopo completo:** F0–F6 ≈ **4,5–7 d** `[modelado]`.

Ordem importa: fazer **F5 depois de F4** evita reescrever a *glue* de `postMessage` a cada
ajuste do *pipeline*. Fazer **F3 antes de F4** evita descobrir o *mismatch* de normalização
já com o modelo carregado.

---

## 4. Armadilhas

Seção distinta de Riscos (seção 6): aqui é técnico e imediato; lá é cronograma e estratégia.

| Armadilha | Mitigação |
|---|---|
| *Threads* WASM exigem `SharedArrayBuffer` → exigem COOP/COEP; vários *hosts* estáticos (GitHub Pages) **não** deixam definir *headers* | Servir por origem própria (Vite/`serve`/Nginx/Cloudflare Pages `_headers`); ter *fallback* `wasm` *single-thread*; checar `crossOriginIsolated` no *boot* |
| `getUserMedia` só funciona em *secure context* | `https://` em produção; `http://localhost` em dev; mensagem clara se `navigator.mediaDevices` for `undefined` |
| Sob `COEP: require-corp`, o `.wasm` do MediaPipe carregado de CDN é **bloqueado** sem `Cross-Origin-Resource-Policy` | Servir os artefatos MediaPipe e ONNX do próprio domínio; ou usar `COEP: credentialless` (Chromium) e aceitar o suporte menor |
| `<video>` espelhado (efeito "espelho") entrega ao modelo uma face invertida | Espelhar só via CSS na exibição; passar ao *canvas* o *frame* não espelhado — ou espelhar consistentemente e aceitar que assimetrias trocam de lado |
| Normalização errada: modelo treinado com média/desvio, código enviando `pixel/255` | Fixar a normalização a partir do *card*/repo do modelo; validar com uma imagem de teste de emoção conhecida |
| Ordem de canais RGB × BGR no *tensor* | Conferir a convenção do `.onnx` (muitos exportados de OpenCV esperam BGR); montar o *tensor* de acordo |
| Layout do *tensor* `NCHW` × `NHWC` | `onnxruntime-web` normalmente espera `[1, C, H, W]`; conferir `session.inputNames` e o *shape* declarado no modelo |
| `getContext('2d')` sem `willReadFrequently: true` → `getImageData` lento a cada *frame* | Passar a *flag* na criação do *context* do *canvas* de trabalho |
| Saída em *logits* tratada como probabilidade no *label* de confiança | Detectar se a soma ≈ 1; aplicar `softmax` só se o modelo não a inclui |
| `setInterval` sofre *throttling* em aba inativa; *frames* acumulam ou somem | `requestVideoFrameCallback`; descartar *frame* se o anterior ainda está em processamento (sem fila) |
| `@vladmandic/face-api` (Trilha A) puxa TFJS como *peer dep* e não recebe *release* há ~2 anos | Fixar versões de `@tensorflow/tfjs-*`; preferir a Trilha B para código novo |
| Modelo FER de vários MB re-baixado a cada visita | `Cache Storage` / *service worker* com *cache-first* para `models/` |
| `argmax` sem limiar mínimo faz o *label* "piscar" entre classes quase empatadas | Exigir confiança ≥ ~0,4 `[modelado]` para trocar o *label*; suavização temporal (F6) |
| `session.run` na *main thread* trava a UI durante a inferência | Web Worker (F5); no mínimo viável, aceitar *jank* e disparar em baixa frequência |
| Nenhuma face no *frame* → recorte vazio → *tensor* de lixo → predição aleatória | Curto-circuito: se a detecção não retorna face, *label* = "nenhum rosto", pular a inferência |

---

## 5. Verificação

Automatizável em CI/host: **quase nada** — esta seção é essencialmente manual e gráfica.
Nenhum *checkbox* nasce marcado; nada foi executado.

**Local, sem navegador (`npm run build` / `lint`):**
- [ ] Os artefatos `.wasm` do `onnxruntime-web` (incl. `ort-wasm-simd-threaded.jsep.wasm`) e o
      `.onnx` do modelo estão no diretório servido — o *bundle* não referencia CDN externa
      para eles.
- [ ] A configuração do servidor estático declara `Cross-Origin-Opener-Policy: same-origin` e
      `Cross-Origin-Embedder-Policy: require-corp`.
- [ ] O mapa de taxonomia expõe exatamente 7 tipos e nunca emite `CONFUSED`/`UNKNOWN`.

**Manual no navegador (Chrome/Edge, `localhost` ou `https`):**
- [ ] A câmera abre e a permissão é pedida **uma vez**; negar a permissão mostra mensagem, não
      tela quebrada.
- [ ] `crossOriginIsolated` é `true` no console (garante que *threads* WASM estão habilitadas).
- [ ] O *label* muda de forma coerente ao alternar expressões (sorrir → `HAPPY`; cara neutra →
      `CALM`).
- [ ] Durante a inferência, o *Performance* não acusa *long tasks* na *main thread* — o
      *pipeline* está no Worker.
- [ ] O EP ativo é `webgpu` onde há suporte e `wasm` onde não há (logar
      `session.handler._ep` ou equivalente / medir latência).
- [ ] *DevTools → Network*: **zero** requisições de imagem ou vídeo saindo da página durante o
      uso (só o *download* inicial de modelo e `.wasm`).

**Cross-browser — declarar como NÃO verificado até rodar:**
- [ ] Firefox: cai em `wasm` sem erro de console e mantém o *label* funcional.
- [ ] Safari (Technology Preview): idem, ou degradação explícita e informada ao usuário.
- [ ] Máquina sem GPU dedicada: `wasm` *single-thread* ainda produz *label*, mesmo que lento.

**Perceptual:**
- [ ] Classificação a ≥ ~10 FPS `[modelado]` numa máquina de desenvolvimento típica.
- [ ] Com a suavização de F6, o *label* não "pisca" entre classes quando a expressão está
      estável.

---

## 6. Riscos

1. **A licença do modelo FER pré-treinado é o caminho crítico, não o código.** Boa parte dos
   pesos FER populares (EmoNet, vários "FER-2013 state of the art") vem com licença de
   pesquisa ou *non-commercial*. Sem um `.onnx` de licença compatível com o uso pretendido, a
   fase F4 trava — e nenhuma quantidade de código a destrava. Levantar isto **antes** de F0.

2. **Acurácia de *benchmark* não se traduz em webcam real.** Números publicados de FER-2013
   (~65–75%) vêm de imagens curadas. Iluminação de escritório, ângulo da webcam, óculos,
   distância e etnias sub-representadas no *dataset* derrubam a confiança real. A suavização
   temporal de F6 **mascara** a instabilidade, não a corrige.

3. **WebGPU ainda não é universal.** Na data deste relatório, só Chrome/Edge têm suporte
   *out-of-box*; Firefox exige *flag*, Safari está em *Technology Preview*. O *fallback*
   `wasm` **tem de ser testado de fato** em cada navegador-alvo, não presumido funcional.

4. **Reconhecimento de emoções carrega restrição regulatória e ética.** O EU AI Act
   classifica sistemas de *emotion recognition* como de risco e restringe seu uso em contexto
   de trabalho e educação. A própria AWS adverte que a saída do Rekognition reflete
   **expressão facial**, não estado emocional interno, e não deve ser usada como tal. Este
   relatório registra a restrição como **condição de uso do produto**, não como detalhe de
   implementação — quem for adotar precisa validar o contexto legal.

5. **Sem *dataset* próprio, não há *fine-tuning*.** Qualquer viés demográfico ou de domínio do
   modelo público entra inalterado no produto. Corrigir exigiria coletar e rotular dados —
   fora do escopo de um projeto client-side de demonstração.

---

## 7. Arquivos tocados

Todos novos — o projeto parte do zero.

| Arquivo | Mudança |
|---|---|
| `index.html` | **novo** — `<video>`, `<canvas>` de exibição, `<div>` do *label*, `<script type="module">` |
| `package.json` | **novo** — dependências `onnxruntime-web`, `@mediapipe/tasks-vision`, *bundler* |
| `src/main.js` | **novo** — *bootstrap*: `getUserMedia`, `requestVideoFrameCallback`, cria o Worker, atualiza o *label* |
| `src/worker.js` | **novo** — *pipeline* no Worker: recebe `ImageBitmap`, chama detecção + FER, devolve `[{ type, confidence }]` |
| `src/fer.js` | **novo** — pré-processamento (recorte, *resize*, normalização, *tensor*) + `InferenceSession` + `session.run` + *argmax* |
| `src/face-detect.js` | **novo** — *wrapper* do MediaPipe Face Detector (`FilesetResolver`, `FaceDetector`) |
| `src/emotions.js` | **novo** — mapa FER-2013 ↔ contrato Rekognition; ordenação por confiança; limiar |
| `models/` | **novo** — `fer.onnx` (quantizado) + `blaze_face_short_range.tflite` do MediaPipe |
| `public/_headers` ou config do servidor | **novo** — COOP/COEP; *cache-first* para `models/` e `*.wasm` |
| `README.md` | **novo** — como rodar em `localhost`, requisitos de navegador, aviso da seção 6 risco 4 |
| `docs/reports/README.md` | **novo** — índice da pasta, com a linha apontando para este relatório |

---

### O que NÃO foi verificado

- **Nenhum número de esforço, tamanho de modelo ou FPS foi medido** — todos marcados
  `[modelado]`. São estimativas de engenharia, não medições.
- **Nenhuma biblioteca foi instalada ou executada.** As versões citadas (`onnxruntime-web`
  ~1.26.0, `@vladmandic/face-api` 1.7.15, `@mediapipe/tasks-vision` ~0.10.x) vêm da
  documentação e dos registros públicos consultados em set/2026, não de um `package-lock.json`
  local.
- **A existência de um `.onnx` de FER com licença aberta e adequada não foi confirmada.** É o
  risco 1 e permanece em aberto.
- **O comportamento de COOP/COEP num *host* concreto não foi testado.** A afirmação de que
  GitHub Pages não permite *headers* customizados é de conhecimento público, não de teste
  feito aqui.
- **Não há hipótese de duplicação a refutar** — o projeto é *greenfield*, não há código
  anterior sobrepondo o escopo.

---

> Nenhum item deste relatório foi executado. O projeto está vazio
> (`/home/luis/Documents/hand-on/toxotes`, `ls -la` → `total 0`), não é repositório git e não
> há *branch* nem *commit* de referência. Toda a análise é prospectiva; a validação está na
> seção 5 e está **integralmente pendente**.
