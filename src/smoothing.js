// F6 — suavizacao temporal. Sem ela o label "pisca" entre classes quase empatadas.
// EMA sobre o vetor de probabilidades: state = alpha*state + (1 - alpha)*amostra.
// alpha alto => mais memoria => mais suave (e mais lento a reagir).
export class EmaSmoother {
  constructor(alpha = 0.6) {
    this.alpha = alpha;
    this.state = null;
  }

  /** @param {number[]} vec @returns {number[]} vetor suavizado (copia) */
  push(vec) {
    if (!this.state || this.state.length !== vec.length) {
      this.state = vec.slice();
      return this.state.slice();
    }
    for (let i = 0; i < vec.length; i += 1) {
      this.state[i] = this.alpha * this.state[i] + (1 - this.alpha) * vec[i];
    }
    return this.state.slice();
  }

  // Chamado quando o rosto some — nao arrastar a distribuicao antiga para quando ele volta.
  reset() {
    this.state = null;
  }
}
