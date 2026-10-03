// Taxonomia e mapeamento para o contrato Amazon Rekognition DetectFaces.
//
// O modelo enet_b0_8_best_afew (EfficientNet-B0, hsemotion-onnx, treinado em
// AffectNet+AFEW+VGAF — ver src/fer.js) tem 8 classes, na ordem abaixo (idx_to_class de
// hsemotion_onnx/facial_emotions.py). Mesmo vocabulário-base do FER+ (herda de FER-2013 +
// `contempt`), só que reordenado.
export const EMOTION_LABELS = [
  'anger',
  'contempt',
  'disgust',
  'fear',
  'happiness',
  'neutral',
  'sadness',
  'surprise',
];

// FER-2013/FER+ -> vocabulário do Amazon Rekognition DetectFaces (Emotions[].Type).
// Mapeamos apenas as 7 classes com correspondência direta ou aproximada.
// `contempt` NÃO tem equivalente no Rekognition e não existe no FER-2013: é descartado,
// e a distribuição restante é renormalizada. Não fabricamos CONFUSED nem UNKNOWN.
const TO_REKOGNITION = {
  happiness: 'HAPPY',
  sadness: 'SAD',
  anger: 'ANGRY',
  surprise: 'SURPRISED',
  disgust: 'DISGUSTED',
  fear: 'FEAR',
  neutral: 'CALM',
};

// Dicionário de localização para Português (PT-BR) e ícones semânticos
export const EMOTION_METADATA = {
  HAPPY:     { pt: 'Feliz',      icon: '😊' },
  SAD:       { pt: 'Triste',     icon: '😢' },
  ANGRY:     { pt: 'Bravo(a)',   icon: '😠' },
  SURPRISED: { pt: 'Surpreso(a)', icon: '😮' },
  DISGUSTED: { pt: 'Desgosto',   icon: '🤢' },
  FEAR:      { pt: 'Medo',       icon: '😨' },
  CALM:      { pt: 'Neutro(a)',  icon: '😌' },
};

/**
 * Converte o vetor de probabilidades do modelo no contrato do Rekognition:
 * `[{ type, confidence }]` ordenado por confiança desc., somando ~1 após renormalizar.
 * @param {number[]} probs - 8 probabilidades na ordem de EMOTION_LABELS
 * @returns {{ type: string, confidence: number }[]}
 */
export function toRekognition(probs) {
  if (!probs) return [];
  const kept = [];
  let total = 0;
  for (let i = 0; i < EMOTION_LABELS.length; i += 1) {
    const type = TO_REKOGNITION[EMOTION_LABELS[i]];
    if (!type) continue; // contempt
    const confidence = probs[i] ?? 0;
    kept.push({ type, confidence });
    total += confidence;
  }
  const norm = total > 0 ? total : 1;
  return kept
    .map((e) => ({ type: e.type, confidence: e.confidence / norm }))
    .sort((a, b) => b.confidence - a.confidence);
}

/**
 * Mapeia o escalar de idade estimada para o contrato Amazon Rekognition DetectFaces (AgeRange).
 * A margem de incerteza padrão é de +/- 4 anos.
 * @param {number|null} rawAge - idade aparente estimada em anos
 * @returns {{ AgeRange: { Low: number, High: number }, estimatedAge: number, formatted: string } | null}
 */
export function toRekognitionAgeRange(rawAge) {
  if (rawAge == null || Number.isNaN(rawAge)) return null;
  const low = Math.max(0, Math.floor(rawAge - 4));
  const high = Math.min(100, Math.ceil(rawAge + 4));
  return {
    AgeRange: { Low: low, High: high },
    estimatedAge: rawAge,
    formatted: `${low}–${high} anos`,
  };
}
