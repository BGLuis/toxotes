// Tracker simples por IoU (intersection-over-union) entre frames — associa cada caixa
// detectada a um id estavel, para que cada rosto tenha seu proprio EmaSmoother e
// stickyLabel em vez de todos os rostos compartilharem um unico estado global (o que
// faria os labels "brigarem" entre si com mais de um rosto na cena).
//
// Sem tracking por ID de verdade (o BlazeFace nao devolve isso), so o melhor casamento
// geometrico frame-a-frame. maxAgeFrames da uma folga curta a oclusoes/flicker de deteccao
// sem trocar o id (e sem reiniciar a suavizacao) por 1-2 frames perdidos.

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

export class FaceTracker {
  constructor({ iouThreshold = 0.3, maxAgeFrames = 3 } = {}) {
    this.iouThreshold = iouThreshold;
    this.maxAgeFrames = maxAgeFrames;
    this.tracks = new Map(); // id -> { box, age }
    this.nextId = 1;
  }

  /**
   * @param {{x:number,y:number,w:number,h:number}[]} boxes - deteccoes do frame atual
   * @returns {{ visible: {id:number, box:object}[], aliveIds: Set<number> }}
   *   `visible` sao os rostos vistos NESTE frame (para desenhar/classificar); `aliveIds`
   *   inclui tambem tracks em periodo de graca (nao vistos, mas ainda nao expirados) —
   *   use para decidir quais estados por-rosto (smoother, stickyLabel) preservar.
   */
  update(boxes) {
    const candidates = [];
    for (const [id, track] of this.tracks) {
      for (let i = 0; i < boxes.length; i += 1) {
        const score = iou(track.box, boxes[i]);
        if (score >= this.iouThreshold) candidates.push({ id, i, score });
      }
    }
    candidates.sort((a, b) => b.score - a.score);

    const usedBoxes = new Set();
    const matchedIds = new Set();
    const visible = [];

    for (const c of candidates) {
      if (usedBoxes.has(c.i) || matchedIds.has(c.id)) continue;
      usedBoxes.add(c.i);
      matchedIds.add(c.id);
      const track = this.tracks.get(c.id);
      track.box = boxes[c.i];
      track.age = 0;
      visible.push({ id: c.id, box: boxes[c.i] });
    }

    for (let i = 0; i < boxes.length; i += 1) {
      if (usedBoxes.has(i)) continue;
      const id = this.nextId++;
      this.tracks.set(id, { box: boxes[i], age: 0 });
      matchedIds.add(id); // nao envelhecer no mesmo update() em que foi criado
      visible.push({ id, box: boxes[i] });
    }

    for (const [id, track] of this.tracks) {
      if (matchedIds.has(id)) continue;
      track.age += 1;
      if (track.age > this.maxAgeFrames) this.tracks.delete(id);
    }

    return { visible, aliveIds: new Set(this.tracks.keys()) };
  }

  reset() {
    this.tracks.clear();
  }
}
