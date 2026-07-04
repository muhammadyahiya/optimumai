// stepper.ts — a tiny clamped stage index, used to drive step-through
// animations (e.g. "Step" button walking through training epochs/stages).

export interface Stepper {
  current(): number;
  next(): void;
  prev(): void;
  reset(): void;
  atEnd(): boolean;
}

/** Create a stepper over `nStages` stages (indices 0..nStages-1, clamped). */
export function createStepper(nStages: number): Stepper {
  const max = Math.max(0, nStages - 1);
  let idx = 0;
  return {
    current(): number {
      return idx;
    },
    next(): void {
      idx = Math.min(max, idx + 1);
    },
    prev(): void {
      idx = Math.max(0, idx - 1);
    },
    reset(): void {
      idx = 0;
    },
    atEnd(): boolean {
      return idx >= max;
    },
  };
}
