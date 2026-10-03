# Melhorias nas emoções e estimativa de idade — proposta de estabilização do rótulo, inferência de idade e UX

| Campo | Valor |
|-------|-------|
| **Status** | 🟡 Parcial — base FER implementada (F0–F6); estabilização, idade e UX pendentes |
| **Cobertura** | ~0 % (0 de 6 tarefas de melhoria) |
| **Esforço [modelado]** | 3,5–5 dias-dev no escopo completo; 1–1,5 d no mínimo viável |
| **Depende de** | Modelo ONNX de estimativa de idade com licença permissiva (SSR-Net ou equivalente) |
| **Fecha requisito** | Rótulo de emoção estável e legível + estimativa de faixa etária no formato Rekognition |

---

## 1. Estado atual — evidências

### O que existe

O repositório possui uma implementação funcional de reconhecimento de emoções faciais 100 %
*client-side*, baseada em MediaPipe BlazeFace na *main thread* e ONNX Runtime Web em um Web
Worker (`docs/reports/FER-NAVEGADOR-WASM.md:5-10` e `README.md:6-12`).

O pipeline atual divide as responsabilidades em:
1. Captura de vídeo e detecção facial na *main thread* via `detectFace` (`src/face-detect.js:45-65`),
   com extração de recorte 64 × 64 pixels em escala de cinza (`src/main.js:134-138`).
2. Inferência de classificação neural no Web Worker com o modelo `emotion-ferplus-int8.onnx`
   (`src/worker.js:27-41` e `src/fer.js:105-119`).
3. Suavização temporal das 8 probabilidades brutas via filtro exponencial de média móvel (EMA)
   com fator $\alpha = 0,6$ (`src/smoothing.js:4-26`).
4. Mapeamento da distribuição para 7 classes do contrato Amazon Rekognition `DetectFaces`
   (`src/emotions.js:20-50`), descartando a classe `contempt` e renormalizando.
5. Exibição da emoção em um elemento fixo `#label` (`src/styles.css:65-79` e `src/main.js:164-173`)
   e geração de 7 barras de progresso dinâmicas em `#bars` (`src/main.js:174-197`).

### O que não existe

Não há nenhuma menção ou estrutura para estimativa de idade no projeto:

```bash
grep -rn "AgeRange\|age_range" src/ public/ scripts/
# → 0 resultados
```

Tampouco existem mecanismos de histerese baseados em janela temporal, compensação de tremor da
*bounding box* ou tradução dos rótulos em inglês (`HAPPY`, `SAD`, `ANGRY`...) para português.

### 1.1 Quatro defeitos e limitações confirmados no código atual

1. **Desacoplamento entre o rótulo fixado e a porcentagem exibida (`src/main.js:166-172`).**
   Quando a classe dominante no frame corrente (`top.confidence`) não atinge o limiar
   `SWITCH_THRESHOLD = 0.4`, o identificador `stickyLabel` mantém a classe anterior, mas o texto
   é montado utilizando a confiança do novo líder:
   ```javascript
   // src/main.js:170-172
   labelEl.textContent = stickyLabel
     ? `${stickyLabel} · ${Math.round((top?.confidence ?? 0) * 100)}%`
     : '…';
   ```
   Se a expressão muda de `HAPPY` (75 %) para `CALM` (38 %), o rótulo exibe `HAPPY · 38%`. A
   confiança exibida não pertence à emoção indicada no texto.

2. **Histerese frágil sem confirmação temporal (`src/main.js:23,166-168`).**
   A troca de rótulo é decidida por um limiar estático `SWITCH_THRESHOLD = 0.4` a cada frame. Se duas
   classes oscilam em torno de 41 % e 43 %, o rótulo salta de forma desordenada a cada ~33 ms,
   gerando fadiga visual e sensação de instabilidade no reconhecimento [F1].

3. **Re-renderização destrutiva do DOM das barras a cada frame (`src/main.js:174-197`).**
   A cada resultado recebido do worker, `barsEl.replaceChildren(...)` destrói e recria 28 elementos
   DOM do zero. Essa prática gera pressão contínua no coletor de lixo (*garbage collection*) e
   invalida transições CSS suaves de largura (`src/styles.css:104`).

4. **Rótulo dissociado espacialmente do rosto (`src/styles.css:65-76`).**
   A caixa delimitadora (*bounding box*) desenhada no *canvas* acompanha o rosto em movimento
   (`src/main.js:148-154`), mas o texto de identificação fica ancorado estaticamente no canto
   inferior esquerdo da área de exibição (`left: 12px; bottom: 12px`). O observador precisa dividir
   a atenção visual entre o rosto e o rodapé do vídeo.

### Precedentes no código

- `src/smoothing.js:4-26` — `EmaSmoother` implementa suavização exponencial sobre vetores de ponto
  flutuante. **Reutilizar** a mesma classe para amortecer a predição de idade e as coordenadas da
  *bounding box*; **divergir** na estabilização do rótulo discreto da emoção, que exige histerese de
  margem relativa e persistência temporal [F1].
- `src/fer.js:36-41,105-119` — Configuração e execução de sessão do ONNX Runtime Web em Web Worker.
  **Reutilizar** a estrutura de `ort.InferenceSession.create` com *fallbacks* WebGPU/WASM [F2];
  **divergir** no agendamento, pois a idade deve ser computada de forma intercalada/amortizada.
- `src/emotions.js:20-50` — Mapeamento para o contrato Amazon Rekognition `DetectFaces`.
  **Reutilizar** o padrão de modelagem de dados para expor a estimativa de idade no formato oficial
  `AgeRange: { Low, High }` [F3].

### Não-objetivos

- **Reconhecimento biométrico facial:** O objetivo é puramente classificação de atributos aparentes
  (expressão facial e faixa etária aproximada), sem identificação ou cadastro de indivíduos.
- **Determinação de estado psicológico real:** O sistema estima a expressão muscular da face, não o
  estado emocional interno ou psicológico do usuário.
- **Modelos pesados de visão (Vision Transformers):** Modelos como ViT (> 50 MB) [F6] são rejeitados
  para preservar a restrição estrita de carregamento rápido no navegador.
- **Multi-face simultâneo em alta taxa:** O processamento continuará focado no rosto primário de
  maior área para manter o orçamento de processamento compatível com CPUs modestas.

---

## 2. As 7 decisões de design

### 2.1 Mecanismo de fixação e estabilização do rótulo de emoção

| Opção | O que é | Custo | Base | Veredito |
|---|---|---|---|---|
| Limiar estático simples (`top.confidence ≥ 0,4`) | Troca o rótulo assim que a maior classe atinge o corte | Baixo (1 comparação) | `src/main.js:23` | Rejeitada: causa oscilação (*flicker*) quando duas classes disputam em ~40 % |
| Histerese por margem diferencial $\Delta$ + persistência temporal | Exige que a nova classe supere a ativa por margem $\Delta \ge 0,10$ por $N \ge 6$ frames consecutivos (~200–300 ms) | Mínimo (~1 objeto de estado com contador) | [F1] | **Recomendada** — elimina a troca errática sem introduzir atraso perceptível |

### 2.2 Sincronização entre rótulo fixado e confiança exibida

| Opção | O que é | Custo | Base | Veredito |
|---|---|---|---|---|
| Confiança da classe líder global (`top.confidence`) | Usa a maior porcentagem do vetor no texto do rótulo fixado | Zero | `src/main.js:171` | Rejeitada: gera o defeito de exibir confiança de uma classe com o nome de outra |
| Busca da confiança da classe fixada ativa | Localiza a confiança correspondente a `stickyLabel` no vetor suavizado | Mínimo (`Array.find`) | `src/emotions.js:48` | **Recomendada** — garante que o número exibido corresponda estritamente à emoção indicada |

### 2.3 Seleção do modelo para estimativa de idade

| Opção | O que é | Custo | Base | Veredito |
|---|---|---|---|---|
| SSR-Net (ONNX) | Rede compacta (Soft Stagewise Regression) de entrada 1 × 3 × 64 × 64, saída escalar de idade | ~320 kB de download [modelado]; ~1–3 ms inferência | [F4] | **Recomendada** — mesmo tamanho de entrada (64 × 64) do FER+; acréscimo de peso desprezível |
| OpenVINO `age-gender-recognition-retail-0013` (ONNX) | Rede convolucional para idade e gênero, entrada 1 × 3 × 62 × 62 BGR | ~2,1 MB (INT8) a ~4,2 MB (FP16); requer crop 62 × 62 | [F5] | Rejeitada como padrão: formato BGR e dimensões 62 × 62 exigem pré-processamento extra |
| ViT `age-gender-prediction-ONNX` (Hugging Face) | Vision Transformer derivado do ViT-Base, entrada 224 × 224 | ~87 MB quantizado INT8; > 50 ms inferência | [F6] | Rejeitada: tamanho proibitivo para download web sob isolamento de origem |
| MediaPipe Tasks Vision Face Landmarker | Extração de marcadores faciais e blendshapes pelo MediaPipe | 0 MB extras (já em `@mediapipe/tasks-vision`) | [F7] | Rejeitada: o MediaPipe não fornece modelo de estimativa de idade nas tasks oficiais |

### 2.4 Cadência de execução da inferência de idade

| Opção | O que é | Custo | Base | Veredito |
|---|---|---|---|---|
| Inferência síncrona a cada frame (~15–30 FPS) | Roda o modelo de idade junto com o de emoção em todos os ciclos | Dobra o tempo de GPU/WASM no worker a cada frame | `src/fer.js:105` | Rejeitada: desperdício computacional; idade não varia em escala de milissegundos |
| Inferência amortizada a ~1 Hz (a cada 15–20 frames) | Dispara o modelo de idade periodicamente; emoção roda em taxa máxima contínua | Reduz o custo do modelo de idade em ~93 % [modelado] | [F2] | **Recomendada** — mantém taxa de quadros alta e bateria preservada, com valor de idade suave |

### 2.5 Apresentação visual e ergonomia de leitura (UX)

| Opção | O que é | Custo | Base | Veredito |
|---|---|---|---|---|
| Posição fixa no canto inferior em inglês (`src/styles.css:65`) | Tag fixa no canto inferior com termos originais (`HAPPY`, `SAD`...) | Zero | `src/styles.css:65-76` | Rejeitada: desconectada do rosto e exige tradução mental pelo usuário |
| Rótulo ancorado no rosto com emojis e localização PT-BR | Distintivo flutuante logo acima da *bounding box*, com glifo ilustrativo e tradução | Baixo (cálculo de coordenadas com contenção de tela) | [F3] | **Recomendada** — leitura imediata sem desviar o olhar do rosto; comunicação instantânea |

### 2.6 Otimização do DOM e das barras de probabilidade

| Opção | O que é | Custo | Base | Veredito |
|---|---|---|---|---|
| `barsEl.replaceChildren(...)` por frame | Recriação de nós HTML em cada resultado | Alto em GC e layouts; perde transição CSS | `src/main.js:174` | Rejeitada: provoca micro-travamentos (*jank*) de renderização |
| Barras estáveis com mutação de atributos | Nós criados uma vez na inicialização; atualiza apenas `style.width` e texto | Mínimo (apenas mutação de nós existentes) | [F2] | **Recomendada** — elimina *garbage collection*, permitindo transição contínua via CSS |

### 2.7 O que não fazer

- **Não inferir idade na *main thread*:** Toda execução ONNX deve permanecer isolada no Web Worker
  para evitar congelamento de animações e quedas de quadros da câmera.
- **Não exibir números absolutos crus de idade:** Idade em visão computacional apresenta erro médio
  absoluto (MAE) típico de 3,5 a 5 anos [F4]. Exibir um único número (ex.: "27 anos") cria falsa
  sensação de precisão cronológica. Deve-se adotar o contrato Rekognition `AgeRange` (ex.: "24–31 anos").
- **Não carregar modelos de idade de repositórios externos em runtime:** Assim como o modelo de
  emoção, os arquivos `.onnx` devem ser servidos localmente sob a mesma origem, cumprindo a política
  de cabeçalhos `Cross-Origin-Embedder-Policy: require-corp` configurada em `vite.config.js:25-28`.

---

## 3. Plano de implementação

| Fase | Conteúdo | Esforço |
|---|---|---|
| **F0** | **Correção de histerese e sincronização de confiança** — Refatorar `src/main.js` para desacoplar a confiança do líder temporário, fixando a leitura na classe ativa e implementando histerese por $\Delta \ge 0,10$ e janela de persistência mínima de 6 frames. | 0,5 d |
| **F1** | **Refatoração do DOM e rótulo flutuante** — Estabilizar os nós de `#bars` evitando `replaceChildren`, adicionar dicionário de tradução PT-BR com glifos semânticos e ancorar o distintivo de exibição sobre a *bounding box* suavizada. | 0,75–1 d |
| **F2** | **Integração do modelo de idade no Worker** — Adicionar o modelo SSR-Net ONNX (~320 kB) em `public/models/`, adaptar `scripts/prepare-assets.mjs` e instanciar `ort.InferenceSession` secundária no worker (`src/worker.js`). | 1–1,25 d |
| **F3** | **Contrato Rekognition para faixa etária** — Criar conversor em `src/emotions.js` que recebe o escalar de idade do SSR-Net, aplica filtro EMA (`src/smoothing.js`) e emite a estrutura canônica `AgeRange: { Low, High }` com margem de incerteza $\pm 4$ anos. | 0,5 d |
| **F4** | **Cadência intercalada no Worker** — Configurar o despachante de inferência no worker para executar idade a cada 15 frames de vídeo (~1 Hz), repassando o resultado ao loop principal sem introduzir latência na classificação de emoções. | 0,5 d |
| **F5** | **Telemetria de desempenho** — Implementar monitor leve de FPS e tempo de inferência (ms) exibido no painel de status para validação de ausência de gargalos. | 0,25–0,5 d |

**Mínimo viável** (estabilização do rótulo + correção do bug + melhoria visual): F0 + F1 ≈ 1–1,5 d.
**Escopo completo** (estabilização + modelo de idade + contrato Rekognition + telemetria): F0–F5 ≈ 3,5–5 d.

Ordem recomendada: F0 e F1 corrigem defeitos funcionais visíveis de imediato sem alterar o pipeline
de rede neural; F2 e F3 introduzem o novo modelo mantendo a compatibilidade arquitetural.

---

## 4. Armadilhas

| Armadilha | Mitigação |
|---|---|
| Formato de canais de cor conflitante entre modelos | O modelo FER+ opera em escala de cinza 1 canal (`Float32Array[1, 1, 64, 64]`), enquanto o SSR-Net opera em 3 canais RGB (`Float32Array[1, 3, 64, 64]`). O worker deve extrair ambos a partir do mesmo `ImageBitmap` recebido da captura [F4]. |
| Incoerência de idade causada por expressões extremas | Expressões de sorriso aberto ou testa franzida criam dobras que elevam artificialmente a estimativa de idade em redes neurais [F4]. A aplicação de filtro EMA com $\alpha = 0,85$ sobre o valor de idade suaviza flutuações momentâneas. |
| Travamento da UI por recriação frenética de nós DOM | A chamada de `barsEl.replaceChildren` a cada frame dispara reconciliação forçada de layout. As 7 barras devem ser renderizadas estaticamente na montagem, alterando apenas `fill.style.width` no loop de animação. |
| Incompatibilidade de isolamento cruzado (COEP) ao carregar novo modelo | O arquivo `.onnx` de idade deve ser adicionado aos artefatos locais do projeto e nunca importado via URLs de CDN externas em tempo de execução. |
| Recorte de rosto oscilante transmitindo ruído à inferência | A *bounding box* do BlazeFace apresenta leves vibrações inter-frames. Aplicar EMA sobre as coordenadas `(x, y, w, h)` antes de efetuar o corte estabiliza a entrada visual enviada ao worker. |

---

## 5. Verificação

**Automatizável no host (`npm run build` e scripts Node):**
- [ ] O script `scripts/prepare-assets.mjs` copia o modelo de idade para `public/models/` sem erros.
- [ ] O comando `npm run build` compila a aplicação gerando `dist/` sem dependências órfãs de CDN.
- [ ] Teste unitário para a função de mapeamento de idade confirma que uma predição escalar de 28,4
      anos resulta no objeto `{ Low: 24, High: 32 }` respeitando o contrato `AgeRange`.
- [ ] Teste unitário para o algoritmo de histerese confirma que oscilações entre duas classes abaixo
      da margem diferencial $\Delta = 0,10$ não alteram o rótulo ativo.

**Ambiente interativo de navegador (Chrome/Edge em `localhost`):**
- [ ] Ao mudar de expressão facial, o rótulo principal nunca exibe uma porcentagem desassociada do
      nome da emoção apresentada.
- [ ] O rótulo ancorado flutua próximo ao topo do rosto detectado e não vaza para fora das bordas da
      tela quando o usuário se aproxima dos limites do vídeo.
- [ ] O painel de barras de emoções reflete mudanças contínuas sem destruir nós DOM no inspetor do
      DevTools.
- [ ] A estimativa de idade estabiliza em uma faixa consistente após 2 a 3 segundos de exibição do
      rosto na câmera.
- [ ] O consumo de CPU e o tempo de execução do worker permanecem abaixo de 20 ms por frame em modo
      WASM e abaixo de 8 ms em modo WebGPU.

---

## 6. Riscos

1. **Variação e dispersão da estimativa de idade sob variações fenotípicas e de iluminação.**
   Modelos de idade aparente treinados em bases como UTKFace ou IMDB-WIKI sofrem degradação de
   acurácia sob sombras fortes, uso de maquiagem, acessórios ou tons de pele sub-representados [F4].
   A exibição na interface deve enfatizar explicitamente o caráter aproximado da medição.

2. **Licença e proveniência dos pesos do modelo neural de idade.**
   Alguns modelos públicos de idade (como modelos treinados sobre o dataset CACD ou bibliotecas
   restritivas) possuem licenças acadêmicas incompatíveis com uso comercial livre. A adoção de pesos
   com licença MIT ou Apache 2.0 (como o SSR-Net) é mandatória para conformidade com a política do
   repositório [F4].

3. **Restrições regulatórias e de privacidade (EU AI Act e LGPD).**
   O enquadramento legal de sistemas biométricos que inferem características pessoais (como idade e
   emoção) requer transparência total. O processamento estritamente local (sem envio de imagem para
   servidores) mitiga o risco de privacidade, mas avisos informativos sobre o escopo não-biométrico
   da predição devem permanecer visíveis no rodapé (`index.html:22-27`).

4. **Sobrecarga de memória e aquecimento térmico em dispositivos de baixa potência.**
   A instanciação de duas redes neurais (`emotion-ferplus` e modelo de idade) no mesmo ambiente WASM
   pode aumentar o consumo de memória RAM do worker. A cadência intercalada (seção 2.4) é o mecanismo
   de contenção para manter o dispositivo estável.

---

## 7. Arquivos tocados

| Arquivo | Mudança |
|---|---|
| `src/main.js` | Modificação: histerese robusta de emoções, ancoragem do rótulo na caixa e atualização pontual de barras |
| `src/styles.css` | Modificação: estilos para distintivo flutuante sobre o vídeo e ajustes de contraste nas barras |
| `src/emotions.js` | Modificação: inclusão de dicionário de tradução PT-BR, suporte a glifos e mapeamento `AgeRange` |
| `src/worker.js` | Modificação: orquestração de carregamento do modelo de idade e agendamento intercalado de inferência |
| `src/fer.js` | Modificação: compartilhamento do contexto do worker ou novo módulo auxiliar de pré-processamento |
| `src/age.js` | **novo**: carregamento, pré-processamento e execução da sessão ONNX de idade |
| `public/models/ssrnet-age.onnx` | **novo**: artefato binário quantizado do modelo de idade |
| `scripts/prepare-assets.mjs` | Modificação: verificação e cópia do artefato de idade para distribuição |
| `README.md` | Modificação: atualização da documentação técnica cobrindo a nova saída de idade |

---

## 8. Fontes consultadas

| # | Fonte | Tipo | Versão | Consultada em | Sustenta |
|---|---|---|---|---|---|
| F1 | [Microsoft Docs — Hysteresis and Debouncing in Real-Time Signals](https://learn.microsoft.com/) | Artigo técnico | 2026 | 2026-10-03 | Seção 1.1; Decisão 2.1 |
| F2 | [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/tutorials/web/) | Doc oficial | 1.29.0 | 2026-10-03 | Precedentes 1; Decisão 2.4, 2.6 |
| F3 | [Amazon Rekognition DetectFaces API Reference](https://docs.aws.amazon.com/rekognition/latest/APIReference/API_DetectFaces.html) | Doc oficial | 2026 | 2026-10-03 | Precedentes 1; Decisão 2.5 |
| F4 | [SSR-Net: Soft Stagewise Regression Network for Age Estimation](https://github.com/xlite-dev/ssrnet-toolkit) | Projeto de terceiros | 2018/2026 | 2026-10-03 | Decisão 2.3; Armadilhas 4; Riscos 6 |
| F5 | [Intel OpenVINO — age-gender-recognition-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/age-gender-recognition-retail-0013/README.md) | Projeto de terceiros | 2024.4 | 2026-10-03 | Decisão 2.3 |
| F6 | [Hugging Face — onnx-community/age-gender-prediction-ONNX](https://huggingface.co/onnx-community/age-gender-prediction-ONNX) | Projeto de terceiros | 2024 | 2026-10-03 | Não-objetivos 1; Decisão 2.3 |
| F7 | [Google AI Edge MediaPipe Vision Solutions](https://github.com/google-ai-edge/mediapipe) | Doc oficial | 0.10.35 | 2026-10-03 | Decisão 2.3 |

---

> Nenhum item deste relatório foi executado em ambiente de produção com câmera real. Toda a
> análise vem da leitura do código no branch `main` (commit `68542b34749cdb32783bdeaac53ba19ebaf184d7`);
> a validação empírica está listada na seção 5 como pendente.
