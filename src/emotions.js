// Taxonomia — ver relatorio secao 2.7.
//
// O modelo enet_b0_8_best_afew (EfficientNet-B0, hsemotion-onnx, treinado em
// AffectNet+AFEW+VGAF — ver src/fer.js) tem 8 classes, na ordem abaixo (idx_to_class de
// hsemotion_onnx/facial_emotions.py). Mesmo vocabulario-base do FER+ (herda de FER-2013 +
// `contempt`), so que reordenado.
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

// FER-2013/FER+ -> vocabulario do Amazon Rekognition DetectFaces (Emotions[].Type).
// Mapeamos apenas as 7 classes com correspondencia direta ou aproximada.
// `contempt` NAO tem equivalente no Rekognition e nao existe no FER-2013: e descartado,
// e a distribuicao restante e renormalizada. Nao fabricamos CONFUSED nem UNKNOWN.
const TO_REKOGNITION = {
  happiness: 'HAPPY',
  sadness: 'SAD',
  anger: 'ANGRY',
  surprise: 'SURPRISED',
  disgust: 'DISGUSTED',
  fear: 'FEAR',
  neutral: 'CALM', // aproximacao: neutral ~= CALM
};

/**
 * Converte o vetor de probabilidades do modelo no contrato do Rekognition:
 * `[{ type, confidence }]` ordenado por confianca desc., somando ~1 apos renormalizar.
 * @param {number[]} probs - 8 probabilidades na ordem de EMOTION_LABELS
 * @returns {{ type: string, confidence: number }[]}
 */
export function toRekognition(probs) {
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
