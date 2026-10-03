// Suavização temporal (EMA).
// Evita oscilação de probabilidades e tremuras visuais (jitter) inter-frames.

export class EmaSmoother {
  constructor(alpha = 0.6) {
    this.alpha = alpha;
    this.state = null;
  }

  /** @param {number[]} vec @returns {number[]} vetor suavizado (cópia) */
  push(vec) {
    if (!vec) return null;
    if (!this.state || this.state.length !== vec.length) {
      this.state = vec.slice();
      return this.state.slice();
    }
    for (let i = 0; i < vec.length; i += 1) {
      this.state[i] = this.alpha * this.state[i] + (1 - this.alpha) * vec[i];
    }
    return this.state.slice();
  }

  reset() {
    this.state = null;
  }
}

/** Suavizador exponencial para valores numéricos escalares (ex.: idade). */
export class ScalarSmoother {
  constructor(alpha = 0.8) {
    this.alpha = alpha;
    this.state = null;
  }

  /** @param {number|null} val @returns {number|null} */
  push(val) {
    if (val == null || Number.isNaN(val)) return this.state;
    if (this.state === null) {
      this.state = val;
      return val;
    }
    this.state = this.alpha * this.state + (1 - this.alpha) * val;
    return this.state;
  }

  reset() {
    this.state = null;
  }
}

/** Suavizador para coordenadas de bounding box { x, y, w, h }. */
export class BoxSmoother {
  constructor(alpha = 0.5) {
    this.alpha = alpha;
    this.state = null;
  }

  /** @param {{ x: number, y: number, w: number, h: number } | null} box */
  push(box) {
    if (!box) {
      this.state = null;
      return null;
    }
    if (!this.state) {
      this.state = { ...box };
      return { ...box };
    }
    this.state.x = this.alpha * this.state.x + (1 - this.alpha) * box.x;
    this.state.y = this.alpha * this.state.y + (1 - this.alpha) * box.y;
    this.state.w = this.alpha * this.state.w + (1 - this.alpha) * box.w;
    this.state.h = this.alpha * this.state.h + (1 - this.alpha) * box.h;
    return {
      x: Math.round(this.state.x),
      y: Math.round(this.state.y),
      w: Math.round(this.state.w),
      h: Math.round(this.state.h),
    };
  }

  reset() {
    this.state = null;
  }
}
