// Taxonomia — ver relatorio secao 2.7.
//
// O modelo emotion-ferplus (ONNX Model Zoo, licenca MIT) tem 8 classes, na ordem abaixo.
// FER+ acrescenta `contempt` ao conjunto do FER-2013.
export const FERPLUS_LABELS = [
  'neutral',
  'happiness',
  'surprise',
  'sadness',
  'anger',
  'disgust',
  'fear',
  'contempt',
];

// FER+ -> vocabulario do Amazon Rekognition DetectFaces (Emotions[].Type).
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
 * Converte o vetor de probabilidades do FER+ no contrato do Rekognition:
 * `[{ type, confidence }]` ordenado por confianca desc., somando ~1 apos renormalizar.
 * @param {number[]} probs - 8 probabilidades na ordem de FERPLUS_LABELS
 * @returns {{ type: string, confidence: number }[]}
 */
export function toRekognition(probs) {
  const kept = [];
  let total = 0;
  for (let i = 0; i < FERPLUS_LABELS.length; i += 1) {
    const type = TO_REKOGNITION[FERPLUS_LABELS[i]];
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
