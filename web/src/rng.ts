// rng.ts — deterministic seedable PRNG so every OptiX widget is reproducible.

/** A seedable pseudo-random generator with a `next()` method in [0, 1). */
export interface PRNG {
  next(): number;
}

/**
 * mulberry32 — small, fast, decent-quality 32-bit PRNG.
 * Same seed always produces the same sequence.
 */
export function mulberry32(seed: number): PRNG {
  let a = seed >>> 0;
  return {
    next(): number {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
  };
}

/** Uniform random number in [min, max). */
export function uniform(rng: PRNG, min = 0, max = 1): number {
  return min + rng.next() * (max - min);
}

/** Standard-normal sample via Box-Muller, driven by a seeded PRNG. */
export function gaussian(rng: PRNG, mean = 0, std = 1): number {
  const u1 = Math.max(rng.next(), 1e-12);
  const u2 = rng.next();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}
